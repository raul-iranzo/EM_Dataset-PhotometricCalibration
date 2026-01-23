import numpy as np
from scipy.spatial.transform import Rotation as R

import utils

from typing import Any, List, Tuple
from nptyping import NDArray


class Base(utils.Optimizable):

    def sample(self,
               T_wc: NDArray[(4, 1), float],
               x_w: NDArray[(4, Any), float]) \
            -> Tuple[NDArray[(1, Any, Any), float], NDArray[(4, Any, Any), float]]:
        ''' Sample light _comming_ to a point in space'''
        assert T_wc.shape == (4, 4), \
            f'`T_wc` attached camera pose must be (4, 4), but {T_wc.shape} encountered.'
        assert x_w.shape[0] == 4, '`x_w` must be homogeneous coordinates'
        assert np.allclose(x_w[3, :], 1), '`x_w` must be point coordinates'
        pass

###############################################################################
##                              COMPLETE CLASS                               ##
###############################################################################


class SpotLightSource(Base):
    '''Spot Light Source (SLS). [Modrzejewski20]
    - Main intensity value: σ_o
    - Light center: P (coordinates XYZ)
    - Normalized principal direction: L(x, P) = (x - P) / ||x - P||
    - Inverse square law: S(x, P) = 1/d², where d = ||x - P||
    - Directional D spread function: R(μ, D, x, P) = e^(-μ(1 - D·L))
    - σ_SLS(x, P) = σ_o · R(μ, D, x, P) · S(x, P) · L(x, P)
    '''

    sigma: float  # main intensity value
    mu: float  # spread factor
    P: NDArray[(4, 1), float]  # light centre in camera reference
    D: NDArray[(4, 1), float]  # principal direction in camera reference

    # New attributes for multi-center area light approximation
    radius: float = 0.0  # radius of the area light
    area_sampling_resolution: int = 0  # level of sampling resolution

    def __init__(self,
                 sigma: float = 1.0,
                 mu: float = 0.0,
                 P: NDArray[(4, 1), float] = np.array(
                     [[0.], [0.], [0.], [1.]]),
                 D: NDArray[(4, 1), float] = np.array(
                     [[0.], [0.], [1.], [0.]]),
                 radius: float = 0.0,
                 area_sampling_resolution: int = 0) -> None:
        assert D.shape == (4, 1), '`D` must be homogeneous direction'
        self.sigma = sigma
        self.mu = mu
        self.P = P
        self.D = D
        self.radius = radius
        self.area_sampling_resolution = area_sampling_resolution
        self._area_sampling_offsets = self._get_area_light_offsets()

    def sample(self,
               T_wc: NDArray[(4, 1), float],
               x_w: NDArray[(4, Any), float]) \
            -> Tuple[NDArray[(1, Any, Any), float], NDArray[(4, Any, Any), float]]:
        super().sample(T_wc, x_w)

        T_cw = np.linalg.inv(T_wc)
        x_c = T_cw @ x_w

        # accumulate contribution from all point lights approximating the area light
        P_all = self.P + self._area_sampling_offsets  # shape (4, N_points)

        vP2x = x_c[:, None, :] - P_all[:, :, None]
        d = np.linalg.norm(vP2x, axis=0, keepdims=True)
        L_x = vP2x / d
        S_x = 1 / (d * d)
        R_x = np.exp(-self.mu * (1 - np.einsum('ik,kjh->ijh', self.D.T, L_x)))

        sigma_SLS = self.sigma * R_x * S_x * L_x

        # return value and direction separately
        value = np.linalg.norm(sigma_SLS, axis=0, keepdims=True)
        w_i =  np.einsum('ik,kjh->ijh', T_wc, -L_x)  # direction towards light in world coordinates
        return value, w_i
    
    def _get_area_light_offsets(self) -> NDArray[(4, Any), float]:
        '''
            Generate point light offsets to approximate area light
            Example for resolution level = 1
                          o---o
                         / \ / \ 
                        o---O---o
                         \ / \ /
                          o---o

            Example for resolution level = 2
                          o---o---o
                         / \ / \ / \
                        o---o---o---o
                       / \ / \ / \ / \
                      o---o---O---o---o
                       \ / \ / \ / \ /
                        o---o---o---o
                         \ / \ / \ /
                          o---o---o
            @return: list of displacement points in homogeneous coordinates
        '''
        if self.radius <= 0.0 or self.area_sampling_resolution < 1:
            # radius is zero or sampling resolution is 0, return single point light
            return np.array([[[0.], [0.], [0.], [0.]]])  # single point light at center
        
        def _get_point(radius: float, angle: float) -> NDArray[(4, 1), float]:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            return np.array([[x], [y], [0.], [0.]])

        HEXAGONE = 6
        offsets = [np.array([[0.], [0.], [0.], [0.]])] # start with center point
        phi = np.linspace(0, 2 * np.pi, HEXAGONE, endpoint=False)
        for level in range(1, self.area_sampling_resolution + 1):
            _radius = self.radius * level / self.area_sampling_resolution
            for i in range(len(phi)):
                sample_point = _get_point(_radius, phi[i])
                prev_sample_point = _get_point(_radius, phi[i-1])
                for j in range(1, level):
                    # interpolate points between current and previous point
                    # level=1 -> 0 interp point
                    # level=2 -> 1 interp points
                    # level=3 -> 2 interp points
                    ratio = j / level
                    interp_point = prev_sample_point * (1 - ratio) + sample_point * ratio
                    offsets.append(interp_point)
                offsets.append(sample_point)
        return np.hstack(offsets)

    def _get_params(self) -> List:
        params = [self.sigma]
        params += [self.mu]
        params += self.P.flatten().tolist()[0:3]
        params += list(utils.cartesian2sphere(self.D))[0:2]
        return params

    def _set_params(self, a: List) -> None:
        self.sigma = a[0]
        self.mu = a[1]
        self.P = np.array(a[2:5] + [1, ]).reshape(4, 1)
        self.D = utils.sphere2cartesian(a[5], a[6])

    def _get_lower_bound(self) -> NDArray:
        return np.array([0,        # sigma
                         0,        # mu
                         -np.inf,  # P_x
                         -np.inf,  # P_y
                         -np.inf,  # P_z
                         0,        # D_theta (elv)
                         0,        # D_phi (azm)
                         ])

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf,   # sigma
                         np.inf,   # mu
                         np.inf,   # P_x
                         np.inf,   # P_y
                         np.inf,   # P_z
                         np.pi/2,  # D_theta (elv)
                         2*np.pi,  # D_phi (azm)
                         ])

    @property
    def T_cl(self) -> NDArray[(4, 4), float]:
        z = np.array([[0], [0], [1], [0]])
        rotvec = np.cross(z[:3, :].T, self.D[:3, :].T).T
        rotvec /= np.linalg.norm(rotvec)
        rotvec *= np.arccos(np.dot(z.T, self.D)[0, 0])
        rot = R.from_rotvec(rotvec.T)
        rotmat = rot.as_matrix()
        T_cl = np.eye(4)
        T_cl[:3, :3] = rotmat
        T_cl[:, 3] = self.P
        return T_cl

###############################################################################
##                          OPTIMIZABLE VARIATIONS                           ##
###############################################################################


class NormalizedSpotLightSource(SpotLightSource):
    ''' SLS with normalized radiance '''

    def _get_params(self) -> List:
        params = [self.mu]
        params += self.P.flatten().tolist()[0:3]
        params += list(utils.cartesian2sphere(self.D))[0:2]
        return params

    def _set_params(self, a: List) -> None:
        self.mu = a[0]
        self.P = np.array(a[1:4] + [1, ]).reshape(4, 1)
        self.D = utils.sphere2cartesian(a[4], a[5])

    def _get_lower_bound(self) -> NDArray:
        return np.array([0,        # mu
                         -np.inf,  # P_x
                         -np.inf,  # P_y
                         -np.inf,  # P_z
                         0,        # D_theta (elv)
                         0,        # D_phi (azm)
                         ])

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf,   # mu
                         np.inf,   # P_x
                         np.inf,   # P_y
                         np.inf,   # P_z
                         np.pi/2,  # D_theta (elv)
                         2*np.pi,  # D_phi (azm)
                         ])


class SpotLightSource2D(SpotLightSource):
    ''' SLS with light center in optical plane '''

    def _get_params(self) -> List:
        params = [self.sigma]
        params += [self.mu]
        params += self.P.flatten().tolist()[0:2]
        params += list(utils.cartesian2sphere(self.D))[0:2]
        return params

    def _set_params(self, a: List) -> None:
        self.sigma = a[0]
        self.mu = a[1]
        self.P = np.array(a[2:4] + [0, 1]).reshape(4, 1)
        self.D = utils.sphere2cartesian(a[4], a[5])

    def _get_lower_bound(self) -> NDArray:
        return np.array([0,        # sigma
                         0,        # mu
                         -np.inf,  # P_x
                         -np.inf,  # P_y
                         0,        # D_theta (elv)
                         0,        # D_phi (azm)
                         ])

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf,   # sigma
                         np.inf,   # mu
                         np.inf,   # P_x
                         np.inf,   # P_y
                         np.pi/2,  # D_theta (elv)
                         2*np.pi,  # D_phi (azm)
                         ])


class NormalizedSpotLightSource2D(SpotLightSource):
    ''' SLS with light center in optical plane and normalized radiance '''

    def _get_params(self) -> List:
        params = [self.mu]
        params += self.P.flatten().tolist()[0:2]
        params += list(utils.cartesian2sphere(self.D))[0:2]
        return params

    def _set_params(self, a: List) -> None:
        self.mu = a[0]
        self.P = np.array(a[1:3] + [0, 1]).reshape(4, 1)
        self.D = utils.sphere2cartesian(a[3], a[4])

    def _get_lower_bound(self) -> NDArray:
        return np.array([0,        # mu
                         -np.inf,  # P_x
                         -np.inf,  # P_y
                         0,        # D_theta (elv)
                         0,        # D_phi (azm)
                         ])

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf,   # mu
                         np.inf,   # P_x
                         np.inf,   # P_y
                         np.pi/2,  # D_theta (elv)
                         2*np.pi,  # D_phi (azm)
                         ])


class FixedSpotLightSource(SpotLightSource):
    ''' Fixed Spot Light Source (FSLS) '''

    def _get_params(self) -> List:
        params = [self.sigma]
        params += [self.mu]
        params += list(utils.cartesian2sphere(self.D))[0:2]
        return params

    def _set_params(self, a: List) -> None:
        self.sigma = a[0]
        self.mu = a[1]
        self.D = utils.sphere2cartesian(a[2], a[3])

    def _get_lower_bound(self) -> NDArray:
        return np.array([0,        # sigma
                         0,        # mu
                         0,        # D_theta (elv)
                         0,        # D_phi (azm)
                         ])

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf,   # sigma
                         np.inf,   # mu
                         np.pi/2,  # D_theta (elv)
                         2*np.pi,  # D_phi (azm)
                         ])


class NormalizedFixedSpotLightSource(SpotLightSource):
    ''' Fixed Spot Light Source (FSLS) with normalized radiance '''

    def _get_params(self) -> List:
        params = [self.mu]
        params += list(utils.cartesian2sphere(self.D))[0:2]
        return params

    def _set_params(self, a: List) -> None:
        self.mu = a[0]
        self.D = utils.sphere2cartesian(a[1], a[2])

    def _get_lower_bound(self) -> NDArray:
        return np.array([0,        # mu
                         0,        # D_theta (elv)
                         0,        # D_phi (azm)
                         ])

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf,   # mu
                         np.pi/2,  # D_theta (elv)
                         2*np.pi,  # D_phi (azm)
                         ])


class ZFixedSpotLightSource(SpotLightSource):
    ''' Fixed Spot Light Source (FSLS) with Z principal direction '''

    def _get_params(self) -> List:
        params = [self.sigma]
        params += [self.mu]
        return params

    def _set_params(self, a: List) -> None:
        self.sigma = a[0]
        self.mu = a[1]

    def _get_lower_bound(self) -> NDArray:
        return np.array([0,        # sigma
                         0,        # mu
                         ])

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf,   # sigma
                         np.inf,   # mu
                         ])


class NormalizedZFixedSpotLightSource(SpotLightSource):
    ''' Fixed Spot Light Source (FSLS) with Z principal direction and normalized radiance '''

    def _get_params(self) -> List:
        params = [self.mu]
        return params

    def _set_params(self, a: List) -> None:
        self.mu = a[0]

    def _get_lower_bound(self) -> NDArray:
        return np.array([0])  # mu

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf])  # mu


class FixedPointLightSource(SpotLightSource):
    ''' Fixed Point Light Source (FPLS) '''

    def _get_params(self) -> List:
        return [self.sigma]

    def _set_params(self, a: List) -> None:
        self.sigma = a[0]

    def _get_lower_bound(self) -> NDArray:
        return np.array([0])  # sigma

    def _get_upper_bound(self) -> NDArray:
        return np.array([np.inf])  # sigma


class NormalizedFixedPointLightSource(SpotLightSource):
    ''' Fixed Point Light Source (FPLS) with normalized radiance '''

    def _get_params(self) -> List:
        return []

    def _set_params(self, a: List) -> None:
        pass

    def _get_lower_bound(self) -> NDArray:
        return np.empty(0)

    def _get_upper_bound(self) -> NDArray:
        return np.empty(0)
