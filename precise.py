"""精密动力学适配层：Orekit 负责力模型，SciPy 只用历史位置估计参数。

本科生先看 main.py；这里隔离坐标变换、力模型和 Java 接口。配置差异见 DYNAMICS.md。
"""
import os
from datetime import date as CalendarDate, timedelta
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

DATA = Path(__file__).resolve().parent / 'data/forces'


@lru_cache(maxsize=1)
def engine():
    import jdk4py
    import orekit_jpype
    import jpype
    if not jpype.isJVMStarted():
        os.environ['JAVA_HOME'] = str(jdk4py.JAVA_HOME)
        orekit_jpype.initVM(vmargs='--enable-native-access=ALL-UNNAMED,-Xmx1024m')
    from java.io import File
    from org.orekit.data import DataContext, DirectoryCrawler
    from org.orekit.frames import FramesFactory
    from org.orekit.time import TimeScalesFactory, AbsoluteDate
    from org.orekit.utils import IERSConventions
    from org.orekit.bodies import CelestialBodyFactory, JPLEphemeridesLoader, OneAxisEllipsoid
    from org.orekit.forces.gravity.potential import GravityFieldFactory, ICGEMFormatReader, FESCnmSnmReader
    DataContext.getDefault().getDataProvidersManager().addProvider(DirectoryCrawler(File(str(DATA))))
    GravityFieldFactory.clearPotentialCoefficientsReaders()
    GravityFieldFactory.addPotentialCoefficientsReader(ICGEMFormatReader('EGM2008-150.gfc', False))
    GravityFieldFactory.clearOceanTidesReaders()
    GravityFieldFactory.addOceanTidesReader(FESCnmSnmReader('eot11a-potential.dat', 1.))
    gravity = GravityFieldFactory.getNormalizedProvider(150, 150)
    for name, kind in [('Sun', JPLEphemeridesLoader.EphemerisType.SUN),
                       ('Moon', JPLEphemeridesLoader.EphemerisType.MOON)]:
        CelestialBodyFactory.clearCelestialBodyLoaders(name)
        CelestialBodyFactory.addCelestialBodyLoader(name, JPLEphemeridesLoader(r'lnx1900\.405', kind))
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
    return dict(itrf=itrf, inertial=FramesFactory.getGCRF(), gravity=gravity,
                sun=CelestialBodyFactory.getSun(), moon=CelestialBodyFactory.getMoon(),
                earth=OneAxisEllipsoid(6378136.3, 1 / 298.257223563, itrf),
                epoch=AbsoluteDate(2000, 1, 1, 12, 0, 0., TimeScalesFactory.getGPS()),
                utc=TimeScalesFactory.getUTC(),
                ut1=TimeScalesFactory.getUT1(IERSConventions.IERS_2010, False))


class PastWeather:
    """只取已结束的日/3h；F10.7 使用后向 81 日均值，预报期冻结到拟合截止时刻。"""
    def __init__(self, cutoff):
        from org.orekit.time import AbsoluteDate
        e = engine()
        day = CalendarDate.fromisoformat(str(cutoff.getComponents(e['utc']).getDate()))
        self.first_day = day - timedelta(days=90)
        self.start = AbsoluteDate(self.first_day.year, self.first_day.month, self.first_day.day, e['utc'])
        self.cutoff = cutoff.durationFrom(self.start)
        self.cutoff_date = cutoff
        rows = []
        for line in (DATA / 'SpaceWeather-All-v1.2.txt').read_text().splitlines():
            if len(line) < 11 or not line[:4].isdigit():
                continue
            row = line.split()
            stamp = CalendarDate(*map(int, row[:3]))
            if self.first_day <= stamp <= day + timedelta(days=1):
                rows.append(row)
        dates = [CalendarDate(*map(int, row[:3])) for row in rows]
        expected = [self.first_day + timedelta(days=k) for k in range(91)]
        if dates[:91] != expected:
            raise ValueError('空间天气未覆盖拟合日期，或日期不连续')
        # CSSI: Kp 列 5:13 以 Kp*10 存储；末三列为 observed F10.7, centered81, trailing81。
        self.flux = np.array([float(r[-3]) for r in rows])
        self.mean = np.array([float(r[-1]) for r in rows])
        self.kp = np.array([[float(v) / 10 for v in r[5:13]] for r in rows]).ravel()

    def elapsed(self, date):
        # 按 UTC 日期索引，跨闰秒时也不能错用前后相邻的 3 h 天气段。
        date = date if date.compareTo(self.cutoff_date) <= 0 else self.cutoff_date
        parts = date.getComponents(engine()['utc'])
        day = CalendarDate.fromisoformat(str(parts.getDate()))
        seconds = (day - self.first_day).days * 86400 + parts.getTime().getSecondsInLocalDay()
        if seconds < 86400:
            raise ValueError('空间天气不覆盖所需历史')
        return seconds

    def getMinDate(self):
        return self.start.shiftedBy(86400.)

    def getMaxDate(self):
        return self.cutoff_date.shiftedBy(31 * 86400.)

    def getInstantFlux(self, date):
        return float(self.flux[int(self.elapsed(date) // 86400) - 1])

    def getMeanFlux(self, date):
        return float(self.mean[int(self.elapsed(date) // 86400) - 1])

    def getThreeHourlyKP(self, date):
        return float(self.kp[int(self.elapsed(date) // 10800) - 1])

    def get24HoursKp(self, date):
        end = int(self.elapsed(date) // 10800)
        return float(self.kp[end - 8:end].mean())


def spacecraft():
    """GRACE-FO 手册的 8 面宏观模型；名义 VVLH 姿态，无未来星敏感器数据。"""
    from java.util import ArrayList
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.forces import BoxAndSolarArraySpacecraft, FixedPanel
    panels = ArrayList()
    definitions = [((1, 0, 0), .9551567, .34, .40), ((-1, 0, 0), .9551567, .34, .40),
        ((0, .766044, -.642787), 3.1554792, .65, .05),
        ((0, -.766044, .642787), .2282913, .34, .40),
        ((0, -.766044, -.642787), 3.1554792, .65, .05),
        ((0, .766044, .642787), .2282913, .34, .40),
        ((0, 0, 1), 6.071112, .12, .68), ((0, 0, -1), 2.167362, .65, .05)]
    for normal, area, absorb, reflect in definitions:
        panels.add(FixedPanel(Vector3D(*map(float, normal)), area, False, 2.2, 0., absorb, reflect))
    return BoxAndSolarArraySpacecraft(panels)


class Arc:
    def __init__(self, observed, position_tolerance=1e-7, max_step=60.):
        self.e = e = engine()
        import jpype
        from org.hipparchus.geometry.euclidean.threed import Vector3D
        from org.orekit.utils import PVCoordinates, IERSConventions
        from org.orekit.attitudes import LofOffset
        from org.orekit.frames import LOFType
        from org.orekit.orbits import CartesianOrbit
        from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction, SolidTides, OceanTides, Relativity
        from org.orekit.forces.drag import DragForce
        from org.orekit.forces.radiation import SolarRadiationPressure
        from org.orekit.models.earth.atmosphere import DTM2000
        from org.orekit.forces.empirical import ParametricAcceleration, HarmonicAccelerationModel
        self.start = e['epoch'].shiftedBy(float(observed[0, 0]))
        self.end = e['epoch'].shiftedBy(float(observed[-1, 0]))
        self.observed = observed
        self.tolerance, self.max_step = position_tolerance, max_step
        self.dates = [e['epoch'].shiftedBy(float(t)) for t in observed[:, 0]]
        self.transforms = [e['itrf'].getTransformTo(e['inertial'], t) for t in self.dates]
        inertial = [tr.transformPVCoordinates(PVCoordinates(Vector3D(*row[1:4]), Vector3D(*row[4:7])))
                    for tr, row in zip(self.transforms, observed)]
        self.initial = np.r_[list(inertial[0].getPosition().toArray()), list(inertial[0].getVelocity().toArray())]
        self.reference = np.array([list(pv.getPosition().toArray()) for pv in inertial])
        orbit = CartesianOrbit(inertial[0], e['inertial'], self.start, e['gravity'].getMu())
        self.period = orbit.getKeplerianPeriod()
        self.attitude = LofOffset(e['inertial'], LOFType.VVLH)
        self.forces = [HolmesFeatherstoneAttractionModel(e['itrf'], e['gravity']),
            ThirdBodyAttraction(e['sun']), ThirdBodyAttraction(e['moon']),
            SolidTides(e['itrf'], e['gravity'].getAe(), e['gravity'].getMu(),
                       e['gravity'].getTideSystem(), IERSConventions.IERS_2010, e['ut1'], e['sun'], e['moon']),
            OceanTides(e['itrf'], e['gravity'].getAe(), e['gravity'].getMu(), 120, 120, IERSConventions.IERS_2010, e['ut1']),
            Relativity(e['gravity'].getMu())]
        self.weather = PastWeather(self.end)
        weather = jpype.JProxy('org.orekit.models.earth.atmosphere.DTM2000InputParameters', inst=self.weather)
        sat = spacecraft()
        drag = DragForce(DTM2000(weather, e['sun'], e['earth']), sat)
        self.drag = sat.getDragParametersDrivers().get(0)
        # 从拟合末端向前分圈，保证最后一个尺度由完整一圈支撑。
        # 起始不足一圈的部分并入首段；预报延续最后一个尺度。
        self.drag_dates = [self.start]
        duration = self.end.durationFrom(self.start)
        for t in sorted(duration - np.arange(1, int(duration / self.period)) * self.period):
            date = self.start.shiftedBy(float(t))
            self.drag.addSpanAtDate(date)
            self.drag_dates.append(date.shiftedBy(.001))
        self.forces += [drag, SolarRadiationPressure(e['sun'], e['earth'], sat)]
        self.empirical = []
        for name, direction in [('S', Vector3D.PLUS_I), ('W', Vector3D.MINUS_J)]:
            for part, phase in [('sin', 0.), ('cos', np.pi / 2)]:
                model = HarmonicAccelerationModel(name + part, self.start, self.period, 1)
                model.getParametersDrivers().get(1).setValue(float(phase))
                self.empirical.append(model.getParametersDrivers().get(0))
                self.forces.append(ParametricAcceleration(direction, self.attitude, model))
        self.scale = np.r_[np.ones(3), np.full(3, .001), np.full(4, 1e-7), np.full(len(self.drag_dates), .1)]

    def propagate(self, parameters, seconds, earth_fixed=False, with_jacobian=False):
        from org.hipparchus.geometry.euclidean.threed import Vector3D
        from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
        from org.orekit.propagation.numerical import NumericalPropagator
        from org.orekit.propagation import SpacecraftState
        from org.orekit.propagation.events import DateDetector
        from org.orekit.propagation.events.handlers import ResetDerivativesOnEvent
        from org.orekit.time import AbsoluteDate
        from org.orekit.orbits import CartesianOrbit, OrbitType
        from org.orekit.utils import PVCoordinates
        import jpype
        p = parameters * self.scale
        for driver, value in zip(self.empirical, p[6:10]):
            driver.setValue(float(value))
        for date, value in zip(self.drag_dates, p[10:]):
            self.drag.setValue(float(1 + value), date)
        state = self.initial + p[:6]
        orbit = CartesianOrbit(PVCoordinates(Vector3D(*state[:3]), Vector3D(*state[3:])),
                               self.e['inertial'], self.start, self.e['gravity'].getMu())
        atol = jpype.JArray(jpype.JDouble)([self.tolerance] * 3 + [self.tolerance / 1000] * 3 + [1e-8])
        rtol = jpype.JArray(jpype.JDouble)([1e-15] * 7)
        integrator = DormandPrince853Integrator(.001, self.max_step, atol, rtol)
        propagator = NumericalPropagator(integrator)
        propagator.setOrbitType(OrbitType.CARTESIAN)
        propagator.setAttitudeProvider(self.attitude)
        propagator.setInitialState(SpacecraftState(orbit).withMass(601.214))
        for force in self.forces:
            propagator.addForceModel(force)
        # 空间天气每 3 h 更新；必须在跳变处断步，不能让积分器跨过去。
        start_day = CalendarDate.fromisoformat(str(self.start.getComponents(self.e['utc']).getDate()))
        end_day = CalendarDate.fromisoformat(str(self.end.getComponents(self.e['utc']).getDate()))
        boundaries = []
        for offset in range((end_day - start_day).days + 1):
            day = start_day + timedelta(days=offset)
            for hour in range(0, 24, 3):
                date = AbsoluteDate(day.year, day.month, day.day, hour, 0, 0., self.e['utc'])
                if self.start.compareTo(date) < 0 and date.compareTo(self.end) <= 0:
                    boundaries.append(date)
        if boundaries:
            detector = DateDetector(jpype.JArray(AbsoluteDate)(boundaries))
            propagator.addEventDetector(detector.withThreshold(1e-9).withHandler(ResetDerivativesOnEvent()))
        if with_jacobian:
            for driver in [*self.empirical, self.drag]:
                driver.setSelected(True)
            harvester = propagator.setupMatricesComputation('stm', None, None)
        generator = propagator.getEphemerisGenerator()
        propagator.propagate(self.start.shiftedBy(float(seconds[-1])))
        ephemeris = generator.getGeneratedEphemeris()
        frame = self.e['itrf'] if earth_fixed else self.e['inertial']
        out, matrices = [], []
        if with_jacobian:
            names = list(harvester.getJacobiansColumnsNames())
            wanted = [d.getNamesSpanMap().get(self.start) for d in self.empirical]
            wanted += [self.drag.getNamesSpanMap().get(d) for d in self.drag_dates]
            indices = [names.index(name) for name in wanted]
        for t in seconds:
            state_at_t = ephemeris.propagate(self.start.shiftedBy(float(t)))
            pv = state_at_t.getPVCoordinates(frame)
            out.append([*pv.getPosition().toArray(), *pv.getVelocity().toArray()])
            if with_jacobian:
                # Orekit 的变分方程给出位置对初态、经验项、阻力尺度的导数。
                initial = np.array(harvester.getStateTransitionMatrix(state_at_t).getData())[:3]
                forces = np.array(harvester.getParametersJacobian(state_at_t).getData())[:3, indices]
                matrices.append(np.column_stack([initial, forces]) * self.scale)
        if with_jacobian:
            return np.asarray(out), np.asarray(matrices).reshape(-1, len(self.scale))
        return np.asarray(out)


def fit_and_predict(observed, future_points):
    print('  Orekit：拟合 8 h 历史（CPU），首次运行需要数分钟……', flush=True)
    arc = Arc(observed)
    times = observed[:, 0] - observed[0, 0]
    spacing = np.diff(times)
    if not len(spacing) or not np.all(spacing == spacing[0]) or spacing[0] not in (10, 60):
        raise ValueError('动力学观测需为完整的 10 秒或 60 秒等间隔序列')
    if times[-1] % 60:
        raise ValueError('拟合弧必须覆盖完整分钟')

    def residual(p):
        return (arc.propagate(p, times)[:, :3] - arc.reference).ravel()

    def jacobian(p):
        return arc.propagate(p, times, with_jacobian=True)[1]

    p0 = np.zeros(len(arc.scale))
    lower = np.r_[np.full(10, -np.inf), np.full(len(arc.drag_dates), -9.9)]
    # 轨道改正很小：先以固定雅可比求解，再在解附近重新线性化核验收敛。
    # 导数由 Orekit 的变分方程计算，不逐个扰动参数反复积分。
    evaluations = 0
    for iteration in range(2):
        print(f'  历史最小二乘，第 {iteration + 1}/2 次线性化', flush=True)
        matrix = jacobian(p0)
        fit = least_squares(residual, p0, jac=lambda p: matrix, bounds=(lower, np.inf), x_scale='jac',
                            max_nfev=30, ftol=1e-6, xtol=1e-4, gtol=1e-4, verbose=1)
        evaluations += fit.nfev
        p0 = fit.x
    if not fit.success:
        raise ValueError('历史拟合未收敛：' + fit.message)
    # 拟合可用 10 s 采样，输出仍为 60 s，方便学生查看和 TimesFM 输入。
    trajectory = arc.propagate(fit.x, np.arange(int(times[-1] / 60) + 1 + future_points) * 60., earth_fixed=True)
    info = dict(gravity='EGM2008 150x150 / Orekit',
        fit_start_gps_seconds=float(observed[0, 0]), fit_end_gps_seconds=float(observed[-1, 0]),
        fit_points=len(observed), observation_step_seconds=float(spacing[0]),
        fit_rms_3d_m=float(np.sqrt(np.mean(fit.fun.reshape(-1, 3)**2) * 3)),
        empirical_m_s2=(fit.x[6:10] * arc.scale[6:10]).tolist(),
        drag_factors=(1 + fit.x[10:] * arc.scale[10:]).tolist(),
        drag_segment_start_gps_seconds=[d.durationFrom(arc.e['epoch']) for d in arc.drag_dates],
        drag_period_seconds=arc.period, initial_state_adjustment=(fit.x[:6] * arc.scale[:6]).tolist(),
        initial_state_frame='GCRF', nominal_mass_kg=601.214,
        parameters_scaled=fit.x.tolist(), solver_status=int(fit.status), function_evaluations=evaluations,
        linearizations=2, jacobian='Orekit variational equations', optimality=float(fit.optimality),
        model='DE405; IERS2010 solid/pole tides; EOT11a 18 waves; DTM2000; 8-panel SRP; 1PR S/W; relativity')
    return trajectory, info
