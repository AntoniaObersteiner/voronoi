#!/usr/bin/python3

import turtle
import time
from random import random

scale = (10, 10)
def goto(P):
    if -10 <= P[0] <= 10 and -10 <= P[1] <= 10:
        x = P[0] * scale[0]
        y = P[1] * scale[1]
        turtle.goto(x, y)

def dist(P, Q = (0, 0)):
    return (
        (P[0] - Q[0])**2 +
        (P[1] - Q[1])**2
    )**.5

class Line:
    def __init__(self, K, L):
        """ implements line satisfying a*x + b*y = d
        with (p, 0) and (0, q) on the line, and (p, -q)
        """
        self.K = K
        self.L = L
        # middle
        self.M = (
            (K[0] + L[0]) / 2,
            (K[1] + L[1]) / 2,
        )
        # normal of line
        n = self.n = (
            L[0] - K[0],
            L[1] - K[1],
        )
        # d of line
        d = self.d = self.n[0]*self.M[0] + self.n[1]*self.M[1]

        a = self.a = n[0]
        b = self.b = n[1]
        self.p = d / a             # point(p, 0) is on line
        self.q = d / b             # point(0, q) is on line
        self.P = (self.p, 0)
        self.Q = (0, self.q)
        self.C = (self.p, -self.q) # vector(p, -q) is along line
        self.c = dist(self.C)
        self.D = dist((a, b))      # norm of normal
        self.N = (a / self.D, b / self.D) # normalized normal

        self.m = (self.M[0] - self.p) / self.p # the paramter for M

        self.update_t(-300)        # line is from P + t*C =: T
        self.update_u( 300)        #           to P + u*C =: U

    def __repr__(self, param = "t"):
        return (
            f"({self.p:3.1f}, 0.0) + {param}*({self.p:3.1f}, {-self.q:3.1f})), "
            f"K = ({self.K[0]:3.1f}, {self.K[1]:3.1f}), "
            f"m = {self.m:3.1f}"
        )

    def __call__(self, t):
        return (self.p + t*self.p, -t*self.q)

    def update_t(self, t):
        if hasattr(self, "t"):
            print(
                f"{self.t = :3.1f} -> {t:3.1f}, "
                f"self.T = ({self.T[0]:3.1f}, {self.T[1]:3.1f}) -> "
                f"({self(t)[0]:3.1f}, {self(t)[1]:3.1f}) "
                f"= {self.__repr__('t')}"
            )
        self.t = t
        self.T = self(t)

    def update_u(self, u):
        if hasattr(self, "u"):
            print(
                f"{self.u = :3.1f} -> {u:3.1f}, "
                f"self.U = ({self.U[0]:3.1f}, {self.U[1]:3.1f}) -> "
                f"({self(u)[0]:3.1f}, {self(u)[1]:3.1f}) "
                f"= {self.__repr__('u')}"
            )
        self.u = u
        self.U = self(u)

    def update(self, r):
        if r < self.m:
            self.update_t(r)
        else:
            self.update_u(r)

    def signed_distance(self, x, y):
        return self.a * x + self.b * y - self.d

    def draw(self):
        turtle.up()
        goto(self.T)
        turtle.down()
        goto(self.U)

    def intersect(self, other):
        a1 =  self.a
        a2 = other.a
        b1 =  self.b
        b2 = other.b
        d1 =  self.d
        d2 = other.d

        #       a₁x       + b₁y = d₁
        #       a₂x       + b₂y = d₂
        δ = a2 / a1
        # (a₁δ-a₂)x + (b₁δ-b₂)y = d₁δ-d₂

        if b1*δ-b2 == 0:
            # no intersection
            return None

        y = (d1*δ-d2) / (b1*δ-b2)
        x = (d1-b1*y) / a1

        I = (x, y)
        return I

    def shorten(self, other, I):
        # where is intersection relative to line anchors
        R =  self.relative(I)
        S = other.relative(I)

        # which direction is that along lines?
        r = R[0] /  self.p
        s = S[0] / other.p

        if (
            self .t <= r <=  self.u and
            other.t <= s <= other.u
        ):
            self .update(r)
            other.update(s)
            return True
        else:
            return False

    def intersect_and_shorten(self, other):
        I = self.intersect(other)
        if I is not None:
            self.shorten(other, I)

    def relative(self, K):
        return (
            K[0] - self.p,
            K[1]
        )

def get_triple_point(P, Q, R):
    # M = (P+Q)/2
    # N = (P-Q)
    # K = (Q+R)/2
    # L = (Q-R)
    # (I-M)*N = 0
    # (I-K)*L = 0
    # I*N = M*N
    # I*L = K*L
    # x*Nx + y*Ny = Mx*Nx + My*Ny =: A
    # x*Lx + y*Ly = Kx*Lx + Ky*Ly =: B
    A = M[0]*N[0] + M[1]*N[1]
    B = K[0]*L[0] + K[1]*L[1]
    # x(Lx-δNx) + y(Ly-δNy) = B-δA
    δ = Lx / Nx
    y = (B-δ*A) / (L[1]-δ*N[1])
    x = (A-y*N[1]) / N[0]
    I = (x, y)

    # the reduction way, just to check
    α = Line(P, Q)
    β = Line(P, R)
    J = α.intersect(β)

    d = dist(I, J)
    if d > .01:
        raise ValueError(f"wrong intersection of {P}, {Q}, {R}. {I = }, {J = }.")

    return I

class Graph:
    def __init__(self, points):
        self.points = points[:]
        self.triples = [
            (i, j, k)
            for i in range(len(self.points))
            for j in range(i)
            for k in range(k)
        ]
        self.triple_points = [
            get_triple_point(
                self.points[i],
                self.points[j],
                self.points[k],
            )
            for i, j, k in self.triples
        ]

turtle.tracer(0, 0)
def main():
    from sys import argv

    def dot(P):
        turtle.up()
        goto(P)
        turtle.dot()

    points = [
        ( 6,  2),
        (-4,  0),
        (-1,  4),
        ( 3, -2),
    ] if len(argv) > 1 else [
        ((random()-.5)*20, (random() - .5)*20)
        for _ in range(10)
    ]
    pairs = [
        (i, j)
        for i in range(len(points))
        for j in range(i)
    ]
    pairs.sort(key = lambda ij: dist(points[ij[0]], points[ij[1]]))
    print(pairs)

    lines = [
        Line(points[i], points[j])
        for i, j in pairs
    ]

    # confine them inside this slightly wonky border
    for line in lines:
        line.intersect_and_shorten(Line((0.1, 0.1), ( 19,   0)))
        line.intersect_and_shorten(Line((0.1, 0.1), (-19,   0)))
        line.intersect_and_shorten(Line((0.1, 0.1), (  0,  19)))
        line.intersect_and_shorten(Line((0.1, 0.1), (  0, -19)))

    def redraw():
        turtle.clear()

        for p, point in enumerate(points):
            dot(point)
            turtle.write(f"  {p}")

        for l in touched:
            lines[l].draw()

        turtle.update()

    touched = set()

    for k in range(len(lines)):
        for l in range(k):
            if len(set(pairs[k]) & set(pairs[l])) == 0:
                continue
            redraw()
            print(f"{k} + {l}: {pairs[k]} + {pairs[l]}")

            if len(argv) > 1:
                input("[ENTER] to continue")

            I = lines[k].intersect(lines[l])
            if I is None:
                continue
            d = dist(I, lines[k].K)
            for p, point in enumerate(points):
                if p in pairs[k]: continue
                if p in pairs[l]: continue
                if dist(I, point) < d:
                    print(f"closer: {p} [{point[0]:3.1f}, {point[1]:3.1f}]")
                    break
            else:
                if lines[k].shorten(lines[l], I):
                    touched |= {k, l}
                    print(f"{touched = }")

    redraw()
    input("[ENTER] to quit")

if __name__ == "__main__":
    main()
