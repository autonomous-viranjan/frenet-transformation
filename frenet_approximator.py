#!/usr/bin/env python3

# Viranjan Bhattacharyya (vbhatta@clemson.edu), 
# EMC2 Lab Clemson University

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from time import time

LANEWIDTH = 3.7

class RoadFrame:
    def __init__(self, road_waypoints_csv: str):
        self.waypoints = np.loadtxt(road_waypoints_csv, delimiter=',')
        self.x_road = self.waypoints[:, 0]
        self.y_road = self.waypoints[:, 1]
        self.s = 0.0
        self.s_dot = 0.0
        self.s_ddot = 0.0
        self.l = 0.0
        self.l_dot = 0.0
        self.l_ddot = 0.0

        self.cspline = sp.interpolate.CubicSpline(self.x_road, self.y_road)
        self.cspline_derivative = self.cspline.derivative()
        self.s_length = 0.0

    def _f_dis(self, point, x_sp):
        """ function to find distance between a road point and a x-parametric point on c-spline """
        y_sp = self.cspline(x_sp)

        return np.linalg.norm(point - np.array([x_sp, y_sp]))
    
    # xy to sl
    def project_point(self, x, y):
        vehicle_point = np.array([x, y])
        self.opt = sp.optimize.minimize_scalar(lambda x_sp: self._f_dis(vehicle_point, x_sp), bounds=(self.x_road[0], self.x_road[-1]))
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
    
    def compute_l(self, x, y):
        vehicle_point = np.array([x, y])
        self.l_dir = (vehicle_point - np.array([self.x_proj, self.y_proj])) / np.linalg.norm((vehicle_point - np.array([self.x_proj, self.y_proj])))
        
        d = self.opt.fun/LANEWIDTH
        sxl = np.cross(self.s_dir, self.l_dir)

        if sxl >= 0.0:
            self.l = d
        else:
            self.l = -d
        self.l += 1.0

    def compute_l_dot(self, vx, vy):
        v = np.array([vx, vy])
        self.l_dot = np.dot(v, self.l_dir) # requires compute l

    def compute_a(self, a, vx, vy):
        v_norm = np.linalg.norm(np.array([vx, vy]))
        v_hat = np.array([vx, vy]) / v_norm if v_norm > 0 else np.array([0., 0.])

        a_net = a * v_hat
        self.s_ddot = np.dot(a_net, self.s_dir)
        self.l_ddot = np.dot(a_net, self.l_dir) # requires compute l dot

    def xy2sl(self, x, y, vx, vy, a):
        self.compute_s(x, y)
        self.compute_s_dot(vx, vy)
        self.compute_l(x, y)
        self.compute_l_dot(vx, vy)
        self.compute_a(a, vx, vy)

        return self.s, self.s_dot, self.s_ddot, self.l, self.l_dot

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
        
    def _arc_len(self, x_cs):
        f = lambda x: np.sqrt(1 + (self.cspline_derivative(x)) ** 2)
        s_len = sp.integrate.quad(f, self.x_road[0], x_cs)
        
        return s_len[0]
    
    def _s_eval(self, s, x_cs):
        return np.linalg.norm(s - self._arc_len(x_cs))
    
    def way_point_res(self):
        distances = []
        for i in range(len(self.waypoints)-1):
            pa = self.waypoints[i, 0:2]
            pb = self.waypoints[i+1, 0:2]
            distances.append(np.linalg.norm(pa - pb))
        self.s_length = np.sum(distances)

        print(f"Max res: {max(distances)}, Min res: {min(distances)}, Avg res: {np.mean(distances)}, Total dis: {np.sum(distances)}")

    def s2x_bounds(self, s):
        dis = 0.0
        i = 0
        while np.linalg.norm(s - dis) > 1.0:
            p1 = np.array([self.x_road[i], self.y_road[i]])
            p2 = np.array([self.x_road[i+1], self.y_road[i+1]])
            dis += np.linalg.norm(p2 - p1)
            i += 1
        return p1[0], p2[0]
    
    def s2x_approximator(self, approximation_res=1e-3):
        """ Since arc integration is not real-time feasible: this function approximator is provided for generating a lookup table """
        self.way_point_res()
        s_series = np.arange(0.0, self.s_length, step=approximation_res)
        s_x = np.zeros((len(s_series), 2))
        for i in range(len(s_series)):
            s_x[i, 0] = s_series[i]
            opt = sp.optimize.minimize_scalar(lambda x_cs: self._s_eval(s_series[i], x_cs), bounds=(self.x_road[0], self.x_road[-1]))
            s_x[i, 1] = opt.x
            print("Generating s to x:", i)
        np.savetxt("s2x_lookup.txt", s_x, fmt="%f", delimiter=",")

    def s2x_lookup(self, s):
        s2x = np.loadtxt("s2x_lookup.txt", delimiter=',')
        e = np.abs(s2x[:, 0] - s)
        i = np.argmin(e)
        print(i)
        # ss = [s2x[i, 0], s2x[i+1, 0]]
        # xx = [s2x[i, 1], s2x[i+1, 1]]        
        # return np.interp(s, ss, xx)
        return s2x[i, 1]
        
    def sl2xy(self, s, v, a, l, ldot, method="lookup"):
        x_l, x_u = self.s2x_bounds(s)
        if method=="crude":
            print("crude")
            x_cs = x_l
        if method=="optim":
            print("optim")
            opt = sp.optimize.minimize_scalar(lambda x_cs:self._s_dis(x_cs, s), bounds=(self.x_road[0], self.x_road[-1]))
            # opt = sp.optimize.minimize_scalar(lambda x_cs: self._s_eval(s, x_cs), bounds=(x_l, x_u))
            x_cs = opt.x
        if method=="lookup":
            print("looking up")
            x_cs = self.s2x_lookup(s)

        print(x_cs)
        y_cs = self.cspline(x_cs)
        slope = self.cspline.derivative()
        si = np.cos(np.arctan2(slope(x_cs), 1))
        sj = np.sin(np.arctan2(slope(x_cs), 1))

        s_hat = np.array([si, sj, 0])
        k_hat = np.array([0, 0, 1])
        l_hat = (l / abs(l)) * np.cross(k_hat, s_hat) if abs(l) > 1e-6 else np.array([0.0, 0.0, 0.0])
        # l_hat = (l / abs(l)) * np.cross(k_hat, s_hat)

        x_ = x_cs + (abs(l) * LANEWIDTH) * np.dot(l_hat, np.array([1, 0, 0]))
        y_ = y_cs + (abs(l) * LANEWIDTH) * np.dot(l_hat, np.array([0, 1, 0]))

        v_net = v * s_hat + ldot * l_hat
        vx_ = np.dot(v_net, np.array([1, 0, 0]))
        vy_ = np.dot(v_net, np.array([0, 1, 0]))

        a_ = a # MPC assumes no lateral acc

        return x_, y_, vx_, vy_, a_

if __name__ == '__main__':
    # Test coordinate transformation
    road_waypoint_csv = "CMI_right_lane_glob.csv"
    coordinate_transform = RoadFrame(road_waypoint_csv)

    generate_lookup = False
    if generate_lookup:
        coordinate_transform.s2x_approximator(approximation_res=0.1)
    
    print("Cartesian to Frenet:")
    # Example current vehicle sensor info
    x = 90.4969499222862
    y = -22.0103088757058
    # x = coordinate_transform.x_road[-1]
    # y = coordinate_transform.y_road[-1]
    vx = 5.0
    vy = -2.0
    a = 0.5

    start = time()
    s, v, a_s, l, ldot = coordinate_transform.xy2sl(x, y, vx, vy, a)
    end = time()
    print("xy2sl time:", end-start)

    print(f"x: {x}, y: {y}, vx: {vx}, vy: {vy}, a: {a}")
    print(f"s: {s}, v: {v}, a: {a_s}, l: {l}, ldot: {ldot}")

    plt.figure(1)
    plt.plot(coordinate_transform.x_road, coordinate_transform.cspline(coordinate_transform.x_road), label="Spline")
    plt.plot(x, y, 'ro', label="vehicle point")
    plt.plot(coordinate_transform.opt.x, coordinate_transform.cspline(coordinate_transform.opt.x), 'go', label="projection point")
    plt.xlim([0, 160])
    plt.ylim([-150, 10])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.show()

    print("\nInverse\n")

    print("Frenet to Cartesian:")

    start = time()
    x, y, vx, vy, a = coordinate_transform.sl2xy(s, v, a_s, l, ldot)
    end = time()
    print("sl2xy sol time:", end-start)

    print(f"s: {s}, v: {v}, a: {a_s}, l: {l}, ldot: {ldot}")
    print(f"x: {x}, y: {y}, vx: {vx}, vy: {vy}, a: {a}")

    plt.figure(2)
    plt.plot(coordinate_transform.x_road, coordinate_transform.cspline(coordinate_transform.x_road), label="Spline")
    plt.plot(x, y, 'ro', label="vehicle point")
    plt.plot(coordinate_transform.opt.x, coordinate_transform.cspline(coordinate_transform.opt.x), 'go', label="projection point")
    plt.xlim([0, 160])
    plt.ylim([-150, 10])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.show()