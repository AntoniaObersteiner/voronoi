#!/usr/bin/python3

import turtle
import time
from random import random
from math import sin, cos, pi

scale = (10, 10)
def goto(P):
    if -40 <= P[0] <= 40 and -40 <= P[1] <= 40:
        x = P[0] * scale[0]
        y = P[1] * scale[1]
        turtle.goto(x, y)

def dist(P, Q = (0, 0)):
    return (
        (P[0] - Q[0])**2 +
        (P[1] - Q[1])**2
    )**.5

class Line:
    def __init__(self, K, L, Q = None, R = None):
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

        # line is from P + t*C =: T
        self.update_t(self.where(Q) if Q is not None else -300)
        #           to P + u*C =: U
        self.update_u(self.where(R) if R is not None else  300)

    def __repr__(self, param = "t"):
        return (
            f"({self.p:3.1f}, 0.0) + {param}*({self.p:3.1f}, {-self.q:3.1f})), "
            f"K = ({self.K[0]:3.1f}, {self.K[1]:3.1f}), "
            f"m = {self.m:3.1f}"
        )

    def __call__(self, t):
        return (self.p + t*self.p, -t*self.q)

    def update_t(self, t):
        self.t = t
        self.T = self(t)

    def update_u(self, u):
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

    def where(self, I):
        # where is intersection relative to line anchors
        R = self.relative(I)
        r = R[0] / self.p
        return r

    def shorten(self, other, I):
        # which direction is I along lines?
        r =  self.where(I)
        s = other.where(I)

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

def avg(P, Q):
    return (
        (P[0] + Q[0]) / 2,
        (P[1] + Q[1]) / 2,
    )
def diff(P, Q):
    return (
        P[0] - Q[0],
        P[1] - Q[1],
    )

def get_triple_point(P, Q, R):
    # M = (P+Q)/2
    # N = (P-Q)
    # K = (Q+R)/2
    # L = (Q-R)
    M =  avg(P, Q)
    N = diff(P, Q)
    K =  avg(Q, R)
    L = diff(Q, R)
    # (I-M)*N = 0
    # (I-K)*L = 0
    # I*N = M*N
    # I*L = K*L
    # x*Nx + y*Ny = Mx*Nx + My*Ny =: A
    # x*Lx + y*Ly = Kx*Lx + Ky*Ly =: B
    # x(Lx-δNx) + y(Ly-δNy) = B-δA
    A = M[0]*N[0] + M[1]*N[1]
    B = K[0]*L[0] + K[1]*L[1]
    if N[0] == 0:
        y = A / N[1]
        if L[0] == 0:
            return None
        x = (B-y*L[1]) / L[0]
    else:
        δ = L[0] / N[0]
        divi = (L[1]-δ*N[1])
        if divi == 0:
            return None
        y = (B-δ*A) / divi
        x = (A-y*N[1]) / N[0]
    I = (x, y)

    return I

def dot(P):
    turtle.up()
    goto(P)
    turtle.dot()

def turtle_init():
    turtle.tracer(0, 0)

def redraw(points, lines, touched = None):
    turtle.clear()

    for p, point in enumerate(points):
        dot(point)
        turtle.write(f"  {p}")

    if touched is None:
        for line in lines:
            line.draw()
    else:
        for l in touched:
            lines[l].draw()

    turtle.update()

def old_main():
    from sys import argv

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

    touched = set()

    for k in range(len(lines)):
        for l in range(k):
            if len(set(pairs[k]) & set(pairs[l])) == 0:
                continue
            redraw(points, lines, touched)
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

    redraw(points, lines, touched)
    input("[ENTER] to quit")

def main():
    from sys import argv

    r1 = 20
    r2 = 30
    speed = 1
    points = ([
        ( 6,  2),
        (-4,  0),
        (-1,  4),
        ( 3, -2),
    ] if len(argv) > 1 else [
        ((random()-.5)*20, (random() - .5)*20)
        for _ in range(20)
    ])
    circles = (
        [(r1*cos(pi*i/180), r1*sin(pi*i/180)) for i in range(2, 360, 19)] +
        [(r2*cos(pi*i/180), r2*sin(pi*i/180)) for i in range(5, 360, 19)]
    )
    while True:
        points = [
            (
                point[0] + speed*(random()-.5),
                point[1] + speed*(random()-.5)
            )
            for point in points
        ]
        lines = get_voronoi_lines(points + circles)

        redraw(points, lines)
    input("[ENTER] to quit")

def get_voronoi_lines(points):
    triples = [
        (p0, p1, p2)
        for p2 in range(len(points))
        for p1 in range(p2)
        for p0 in range(p1)
    ]
    centers = {
        (p0, p1, p2): center
        for p0, p1, p2 in triples
        if (center := get_triple_point(points[p0], points[p1], points[p2])) is not None
    }
    distances = {
        π: dist(center, points[π[0]])
        for π, center in centers.items()
    }
    relevants = [
        π for π in centers.keys()
        if all(
            distances[π] <= dist(points[p], centers[π])
            or p in π
            for p in range(len(points))
        )
    ]
    edges = [
        (π, ρ)
        for π in relevants
        for ρ in relevants
        if len(set(π) & set(ρ)) == 2
    ]

    lines = []
    for π, ρ in edges:
        p, o = set(π) & set(ρ)

        try:
            lines.append(Line(points[p], points[o], centers[π], centers[ρ]))
        except ZeroDivisionError:
            pass

    return lines

if __name__ == "__main__":
    turtle_init()
    #old_main()
    main()
