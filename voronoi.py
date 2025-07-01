#!/usr/bin/python3

import turtle
from time import sleep, time
from random import random
from math import sin, cos, pi, ceil, floor
from collections import defaultdict
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--number-of-points", "-p",
    default = 20,
    type = int,
    help = "how many points to add in the center",
)
argparser.add_argument(
    "--speed", "-s",
    default = 1.0,
    type = float,
    help = "how fast the points move",
)
argparser.add_argument(
    "--granularity", "-g",
    default = 5,
    type = int,
    help = "size of the grid cells that are used for optimization",
)
argparser.add_argument(
    "--rounds", "-r",
    default = 0,
    type = int,
    help = "how many steps to run before resetting, 0 is inifinity",
)
argparser.add_argument(
    "--scale", "-S",
    default = 10,
    type = int,
    help = "how to scale the drawing",
)
argparser.add_argument(
    "--border", "-B",
    default = 40,
    type = int,
    help = "only draw what is inside this in x and y: [-B, B]",
)
argparser.add_argument(
    "--inner-radius", "-I",
    default = 20,
    type = int,
    help = "the inner circle of dots created to have a nice border",
)
argparser.add_argument(
    "--outer-radius", "-O",
    default = 30,
    type = int,
    help = "the outer circle of dots created to have a nice border",
)
argparser.add_argument(
    "--inner-phase",
    default = 4,
    type = int,
    help = "the angle of the inner circle of dots created to have a nice border",
)
argparser.add_argument(
    "--outer-phase",
    default = 8,
    type = int,
    help = "the angle of the outer circle of dots created to have a nice border",
)
argparser.add_argument(
    "--inner-step",
    default = 19,
    type = int,
    help = "the angle step of the inner circle of dots created to have a nice border",
)
argparser.add_argument(
    "--outer-step",
    default = 19,
    type = int,
    help = "the angle step of the outer circle of dots created to have a nice border",
)
argparser.add_argument(
    "--numbers", "-n",
    action = "store_const",
    const = True,
    default = True,
    help = "draw numbers for the points",
)
argparser.add_argument(
    "--no-numbers", "-N",
    action = "store_const",
    const = False,
    dest = "numbers",
    help = "don't draw numbers for the points",
)
argparser.add_argument(
    "--points",
    action = "store_const",
    const = True,
    default = True,
    help = "draw points for the points",
)
argparser.add_argument(
    "--no-points", "-P",
    action = "store_const",
    const = False,
    dest = "points",
    help = "don't draw points for the points",
)
argparser.add_argument(
    "--ask-to-continue", "-C",
    action = "store_const",
    const = True,
    default = False,
    help = "ask for input in each round",
)
argparser.add_argument(
    "--dont-ask-to-continue",
    action = "store_const",
    const = False,
    dest = "ask_to_continue",
    help = "don't ask to for input in each round",
)
argparser.add_argument(
    "--ask-to-quit", "-Q",
    action = "store_const",
    const = True,
    default = True,
    help = "ask for input at the end",
)
argparser.add_argument(
    "--dont-ask-to-quit", "-q",
    action = "store_const",
    const = False,
    dest = "ask_to_quit",
    help = "don't ask to for input at the end",
)

if __name__ == "__main__":
    args = argparser.parse_args()

def goto(P):
    if -40 <= P[0] <= 40 and -40 <= P[1] <= 40:
        x = P[0] * args.scale
        y = P[1] * args.scale
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
        r = (I[0] - self.p) / self.p
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

def get_triple_point(P, Q, R):
    # M = (P+Q)/2
    # N = (P-Q)
    # K = (Q+R)/2
    # L = (Q-R)
    M = (
        (P[0] + Q[0]) / 2,
        (P[1] + Q[1]) / 2,
    )
    N = (
        P[0] - Q[0],
        P[1] - Q[1],
    )
    K = (
        (R[0] + Q[0]) / 2,
        (R[1] + Q[1]) / 2,
    )
    L = (
        R[0] - Q[0],
        R[1] - Q[1],
    )
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
        if args.numbers:
            turtle.write(f"  {p}")

    if touched is None:
        for line in lines:
            line.draw()
    else:
        for l in touched:
            lines[l].draw()

    turtle.update()

def infinite_loop():
    i = 0
    while True:
        yield i
        i += 1

def main():
    points = [
        ((random()-.5)*20, (random() - .5)*20)
        for _ in range(args.number_of_points)
    ]
    def cis(radius, degree):
        radians = pi * degree / 180
        return (radius * cos(radians), radius * sin(radians))

    # args.inner_phase values that don't produce artefacts: 4, 14, 23, 31, 33
    circles = (
        [cis(args.inner_radius, i) for i in range(args.inner_phase, 360, args.inner_step)] +
        [cis(args.outer_radius, i) for i in range(args.outer_phase, 360, args.outer_step)]
    )
    stamp = time()
    grid = Grid(points + circles, args.granularity)

    loop = range(args.rounds) if args.rounds else infinite_loop()
    for i in loop:
        for p, point in enumerate(points):
            grid.points[p] = (
                point[0] + args.speed*(random()-.5),
                point[1] + args.speed*(random()-.5)
            )
            grid.update(p, point, points[p])
        lines = get_voronoi_lines(grid)

        if use_turtle:
            points_to_draw = grid.points[:len(points)] if args.points else []
            redraw(points_to_draw, lines)
        print(f"{time() - stamp = }")
        stamp = time()
        if args.ask_to_continue:
            input("[ENTER] to continue")

    if args.ask_to_quit:
        input("[ENTER] to quit")

class Grid:
    def round(self, point):
        return (
            int(point[0] / self.side),
            int(point[1] / self.side),
        )

    def update(self, p, old_pos, new_pos):
        old_ij = self.round(old_pos)
        new_ij = self.round(new_pos)
        if old_ij == new_ij:
            return

        self.data[old_ij].remove(p)
        self.data[new_ij].append(p)

        if len(self.data[old_ij]) == 0:
            del self.data[old_ij]

    def __init__(self, points, side = 3):
        self.side = side
        self.points = points
        self.data = defaultdict(lambda: [])
        self.box = ((10000, 10000), (-10000, -10000))
        for p, point in enumerate(points):
            ij = self.round(point)
            self.data[ij].append(p)
            self.box = (
                (   min(self.box[0][0], ij[0]),
                    min(self.box[0][1], ij[1]),  ),
                (   max(self.box[1][0], ij[0]),
                    max(self.box[1][1], ij[1]),  ),
            )
        self.size = max(
            self.box[1][0] - self.box[0][0],
            self.box[1][1] - self.box[0][1],
        )

    def around(self, center, distance):
        I, J = self.round(center)
        D = ceil(distance / self.side) + 1
        if D > self.size:
            for p in range(len(self.points)):
                yield p
            return
        for d in range(0, D):
            for i in range(0, d + 1):
                keys = {
                    (I+i, J+d),
                    (I-i, J+d),
                    (I+i, J-d),
                    (I-i, J-d),
                    (I+d, J+i),
                    (I-d, J+i),
                    (I+d, J-i),
                    (I-d, J-i),
                }
                for key in keys:
                    if key not in self.data.keys():
                        continue
                    for p in self.data[key]:
                        yield p

def get_voronoi_lines(grid):
    points = grid.points
    triples = [
        (p0, p1, p2)
        for p2 in range(len(points))
        for p1 in range(p2)
        for p0 in range(p1)
    ]
    centers = {}
    distances = {}
    relevants = []
    for π in triples:
        center = get_triple_point(points[π[0]], points[π[1]], points[π[2]])
        if center is not None:
            centers[π] = center
            distances[π] = dist(center, points[π[0]])

            is_relevant = True
            for p in grid.around(centers[π], distances[π]):
                if p in π:
                    continue
                if distances[π] > dist(points[p], centers[π]):
                    is_relevant = False
                    break
            if is_relevant:
                relevants.append(π)

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
    try:
        turtle_init()
        use_turtle = True
    except:
        pass
        use_turtle = False

    from sys import argv

    for _ in range(10):
        print("=================")
        main()

