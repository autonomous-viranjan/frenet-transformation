#!/usr/bin/env python3

# Viranjan Bhattacharyya (vbhatta@clemson.edu)

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class RoadFrame:
    def __init__(self, road_waypoints_csv: str):
        self.waypoints = np.loadtxt(road_waypoints_csv, delimiter=',')
        self.x_road = self.waypoints[:, 0]
        self.y_road = self.waypoints[:, 1]
        self.s = 0.0
        self.s_dot = 0.0
        self.a = 0.0
        self.l = 0.0
        self.l_dot = 0.0
        self.cspline = sp.interpolate.CubicSpline(self.x_road, self.y_road)        

    # xy to sl
    def f_dis(self, point, x_sp):
        """ function to find distance between a road point and a x-parametric point on c-spline """
        y_sp = self.cspline(x_sp)

        return np.linalg.norm(point - np.array([x_sp, y_sp]))
    
    def project_point(self, x, y):
        vehicle_point = np.array([x, y])
        self.opt = sp.optimize.minimize_scalar(lambda x_sp: self.f_dis(vehicle_point, x_sp), bounds=(self.x_road[0], self.x_road[-1]))
        x_proj = self.opt.x
        y_proj = self.cspline(x_proj)

        return x_proj, y_proj
    
    def compute_s(self, x, y):
        """ computes s till projected point on cspline & index of closest road waypoint (for other vector calcs) """
        self.x_proj, self.y_proj = self.project_point(x, y)
        s = 0.0
        i = 0
        try:
            while self.x_road[i+1] <= self.x_proj:
                s += np.linalg.norm(np.array([self.x_road[i], self.cspline(self.x_road[i])]) - np.array([self.x_road[i+1], self.cspline(self.x_road[i+1])]))
                i += 1
            s += np.linalg.norm(np.array([self.x_road[i], self.cspline(self.x_road[i])]) - np.array([self.x_proj, self.cspline(self.x_proj)]))
            self.s = s
            self.idx_road = i
        except:
            raise Exception("Out of bounds")
        
    def compute_s_dot(self, vx, vy):
        v = np.array([vx, vy])
        slope = self.cspline.derivative()
        theta_s = np.arctan2(slope(self.x_proj), 1)
        self.s_dir = np.array([np.cos(theta_s), np.sin(theta_s)])
            
        self.s_dot = np.dot(v, self.s_dir)

    def compute_a(self, a, vx, vy):
        v_hat = np.array([vx, vy]) / np.linalg.norm(np.array([vx, vy]))
        a_net = a * v_hat
        self.a = np.dot(a_net, self.s_dir)
    
    def compute_l(self, x, y):
        vehicle_point = np.array([x, y])
        self.l_dir = (vehicle_point - np.array([self.x_proj, self.y_proj])) / np.linalg.norm((vehicle_point - np.array([self.x_proj, self.y_proj])))
        LANEWIDTH = 3.7
        d = self.opt.fun/LANEWIDTH
        sxl = np.cross(self.s_dir, self.l_dir)

        if sxl >= 0.0:
            self.l = d
        else:
            self.l = -d

    def compute_l_dot(self, vx, vy):
        v = np.array([vx, vy])
        self.l_dot = np.dot(v, self.l_dir)

    def xy2sl(self, x, y, vx, vy, a):
        self.compute_s(x, y)
        self.compute_s_dot(vx, vy)
        self.compute_a(a, vx, vy)
        self.compute_l(x, y)
        self.compute_l_dot(vx, vy)

        return self.s, self.s_dot, self.a, self.l, self.l_dot

    # sl to xy
    def _s_dis(self, x_cs, s):
        """ function to compute distance between s on spline and parametrized point cspline(x_cs) """
        sdis = 0.0
        i = 0
        try:
            while self.x_road[i+1] <= x_cs:
                straight = np.linalg.norm(np.array([self.x_road[0], self.cspline(self.x_road[0])]) - np.array([self.x_road[i], self.cspline(self.x_road[i])]))
                sdis += np.linalg.norm(np.array([self.x_road[i], self.cspline(self.x_road[i])]) - np.array([self.x_road[i+1], self.cspline(self.x_road[i+1])]))
                i += 1
            sdis += np.linalg.norm(np.array([self.x_road[i], self.cspline(self.x_road[i])]) - np.array([x_cs, self.cspline(x_cs)]))
            return np.abs(sdis - s)
        except:
            raise Exception("Out of bounds")
        
    def sl2xy(self, s, v, l, ldot):
        opt = sp.optimize.minimize_scalar(lambda x_cs:self._s_dis(x_cs, s), bounds=(self.x_road[0], self.x_road[-1]))
        x_cs = opt.x
        y_cs = self.cspline(opt.x)
        slope = self.cspline.derivative()
        si = np.cos(np.arctan2(slope(x_cs), 1))
        sj = np.sin(np.arctan2(slope(x_cs), 1))

        s_hat = np.array([si, sj, 0])
        k_hat = np.array([0, 0, 1])
        l_hat = np.cross(k_hat, s_hat)

        LANEWIDTH = 3.7
        x_ = x_cs + LANEWIDTH * l * np.dot(l_hat, np.array([1, 0, 0]))
        y_ = y_cs + LANEWIDTH * l * np.dot(l_hat, np.array([0, 1, 0]))

        v_net = v * s_hat + ldot * l_hat
        vx_ = np.dot(v_net, np.array([1, 0, 0]))
        vy_ = np.dot(v_net, np.array([0, 1, 0]))

        return x_, y_, vx_, vy_

if __name__ == '__main__':
    # Test coordinate transformation
    road_waypoint_csv = "CMI_right_lane_glob.csv"
    coordinate_transform = RoadFrame(road_waypoint_csv)
    
    print("Cartesian to Frenet:")
    # Example current vehicle sensor info
    x = 90.4969499222862
    y = -22.0103088757058
    vx = 5.0
    vy = -2.0
    a = 0.5

    s, v, a, l, ldot = coordinate_transform.xy2sl(x, y, vx, vy, a)

    print(f"s: {s}, v: {v}, a: {a}, l: {l}, ldot: {ldot}")

    print("Frenet to Cartesian:")
    # Example s-l coordinates
    s = 75.69221462798248
    v = 4.9494168855983816
    a = 0.45954182117520115
    l = 0.9905090025158598
    ldot = 2.1220916207705574

    x, y, vx, vy = coordinate_transform.sl2xy(s, v, l, ldot)

    print(f"x: {x}, y: {y}, vx: {vx}, vy: {vy}")

    plt.plot(coordinate_transform.x_road, coordinate_transform.cspline(coordinate_transform.x_road), label="Spline")
    plt.plot(x, y, 'ro', label="vehicle point")
    plt.plot(coordinate_transform.opt.x, coordinate_transform.cspline(coordinate_transform.opt.x), 'go', label="projection point")
    plt.xlim([0, 160])
    plt.ylim([-150, 10])
    plt.legend()
    plt.show()
