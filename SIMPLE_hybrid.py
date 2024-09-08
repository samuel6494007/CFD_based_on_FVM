# SIMPLE, hybrid
import numpy as np
import matplotlib.pyplot as plt

# 원래 행렬 및 2차원 배열은
#           i=0   00   01   02
#           i=1   10   11   12
#           i=2   20   21   22
#                j=0  j=1   j=2
# 인데 책에서는
#           j=0   00   01   02
#           j=1   10   11   12
#           j=2   20   21   22
#                i=0  i=1   i=2
#
# 로 표현하여 a(J, I)로 표현하기로 함
# 따라서 e는 a(J, I+1)로 표현
# I: x  J: y
# a(J, I) -> 그대로 하되, 책에서의 I, J 연산은 그대로 진행
# J: ---  I: |

# 상수 정의
mu = 0.001003

# 해석 변수
alpha = 0.2  # 하양이완 계수  알파가 1이 아닐 때 유동이 이상해짐 -> u 역전 및 y 교란
r = 5  # 반복 계수

# 검사체적 설정
m = 41
n = 41

x = 0.01
y = 0.005
dx = x/(n-1)  # [m]
dy = y/(m-1)  # [m]

# 격자 생성
# 압력/속도
p = np.zeros((m, n))
u = np.zeros((m, n - 1))
v = np.zeros((m - 1, n))

# 추정 압력/속도 *
p0 = np.zeros((m, n))
u0 = np.zeros((m, n - 1))
v0 = np.zeros((m - 1, n))

# 수정 압력/속도 '
p1 = np.zeros((m, n))
u1 = np.zeros((m, n - 1))
v1 = np.zeros((m - 1, n))

# 초기조건
def initial_condition():
    p0[:, :] = 101325
    u0[:, :] = 1
    v0[:, :] = 0

# 경계조건
u_inlet = 1
v_inlet = 0
p_outlet = 101325
v_top = 0
v_bottom = 0
def boundary_condition():

    # inlet
    u0[:, 0] = u_inlet
    v0[:, 0] = v_inlet

    # outlet
    p0[:, n - 2] = p_outlet

    # top wall
    v0[0, :] = v_top

    # bottom wall
    v0[m - 2, :] = v_bottom

# 벽 경계조건 소스항
#Sp_wall_u = -mu * dx * 2/dy
Sp_wall_u = 0
Sp_wall_v = -mu * dy * 2/dx


# 기체 물성 변수
def rho(J, I):
    return 998


# 하이브리드 차분법
def F_w(J, I):
    return (rho(J, I) + rho(J, I - 1))/2 * u0[J, I - 1] * dy

def F_e(J, I):
    return (rho(J, I) + rho(J, I + 1))/2 * u0[J, I] * dy

def F_s(J, I):
    return (rho(J, I) + rho(J + 1, I))/2 * v0[J, I] * dx

def F_n(J, I):
    return (rho(J, I) + rho(J - 1, I))/2 * v0[J - 1, I] * dx

def deltaF(J, I):
    return F_e(J, I) - F_w(J, I) + F_n(J, I) - F_s(J, I)

def D_w(J, I, Gamma):
    return Gamma/dx

def D_e(J, I, Gamma):
    return Gamma/dx

def D_s(J, I, Gamma):
    return Gamma/dy

def D_n(J, I, Gamma):
    return Gamma/dy

# 운동량 방정식 F, D, 유속 기준
def Fu_w(J, I):
    if I == 0:
        return ((rho(J, I) + rho(J, I))/2 * u0[J, I] + (rho(J, I) + rho(J, I + 1))/2 * u0[J, I])/2
    else:
        return ((rho(J, I) + rho(J, I - 1))/2 * u0[J, I - 1] + (rho(J, I) + rho(J, I + 1))/2 * u0[J, I])/2

def Fu_e(J, I):
    if I == n-2:
        return ((rho(J, I) + rho(J, I + 1))/2*u0[J, I] + (rho(J, I + 1) + rho(J, I + 1))/2*u0[J, I])/2
    else:
        return ((rho(J, I) + rho(J, I + 1))/2*u0[J, I] + (rho(J, I + 1) + rho(J, I + 2))/2*u0[J, I + 1])/2

def Fu_s(J, I):
    return ((rho(J, I) + rho(J + 1, I))/2 * v0[J, I] + (rho(J, I + 1) + rho(J + 1, I + 1))/2*v0[J, I + 1])/2

def Fu_n(J, I):
    return ((rho(J, I) + rho(J - 1, I))/2*v0[J - 1, I] + (rho(J, I + 1) + rho(J - 1, I + 1))/2*v0[J - 1, I + 1])/2

def deltaFu(J, I):
    return Fu_e(J, I) - Fu_w(J, I) + Fu_n(J, I) - Fu_s(J, I)

def Fv_w(J, I):
    return ((rho(J, I) + rho(J, I - 1))/2*u0[J, I - 1] + (rho(J + 1, I) + rho(J + 1, I - 1))/2*u0[J + 1, I - 1])/2

def Fv_e(J, I):
    return ((rho(J, I) + rho(J, I + 1))/2*u0[J, I] + (rho(J + 1, I) + rho(J + 1, I + 1))/2*u0[J + 1, I])/2

def Fv_s(J, I):
    if J==m-2:
        return ((rho(J, I) + rho(J + 1, I))/2*v0[J, I] + (rho(J + 1, I) + rho(J + 1, I))/2*v0[J, I])/2
    else:
        return ((rho(J, I) + rho(J + 1, I))/2*v0[J, I] + (rho(J + 1, I) + rho(J + 2, I))/2*v0[J + 1, I])/2

def Fv_n(J, I):
    if J==0:
        return ((rho(J + 1, I) + rho(J, I))/2*v0[J, I] + (rho(J, I) + rho(J, I))/2*v0[J, I])/2
    else:
        return ((rho(J + 1, I) + rho(J, I))/2*v0[J, I] + (rho(J, I) + rho(J - 1, I))/2*v0[J - 1, I])/2

def deltaFv(J, I):
    return Fv_e(J, I) - Fv_w(J, I) + Fv_n(J, I) - Fv_s(J, I)

# 하이브리드 차분법, u
def ahu_w(J, I):
    return max(Fu_w(J, I), D_w(J, I, mu) + Fu_w(J, I)/2, 0)

def ahu_e(J, I):
    return max(-Fu_e(J, I), D_e(J, I, mu) - Fu_e(J, I)/2, 0)

def ahu_s(J, I):
    return max(Fu_s(J, I), D_s(J, I, mu) + Fu_s(J, I)/2, 0)

def ahu_n(J, I):
    return max(-Fu_n(J, I), D_n(J, I, mu) - Fu_n(J, I)/2, 0)

def ahu_p(J, I):
    return ahu_w(J, I) + ahu_e(J, I) + ahu_s(J, I) + ahu_n(J, I) + deltaFu(J, I)

# 하이브리드 차분법, v
def ahv_w(J, I):
    return max(Fv_w(J, I), D_w(J, I, mu) + Fv_w(J, I)/2, 0)

def ahv_e(J, I):
    return max(-Fv_e(J, I), D_e(J, I, mu) - Fv_e(J, I)/2, 0)

def ahv_s(J, I):
    return max(Fv_s(J, I), D_s(J, I, mu) + Fv_s(J, I)/2, 0)

def ahv_n(J, I):
    return max(-Fv_n(J, I), D_n(J, I, mu) - Fv_n(J, I)/2, 0)

def ahv_p(J, I):
    return ahv_w(J, I) + ahv_e(J, I) + ahv_s(J, I) + ahv_n(J, I) + deltaFv(J, I)

# 압력 계산 변수/계수
def ap_w(J, I):
    return ((rho(J, I) + rho(J, I - 1))/2)*dy*dy/ahu_p(J, I - 1)

def ap_e(J, I):
    return ((rho(J, I) + rho(J, I + 1))/2)*dy*dy/ahu_p(J, I)

def ap_s(J, I):
    return ((rho(J, I) + rho(J + 1, I))/2)*dx*dx/ahv_p(J, I)

def ap_n(J, I):
    return ((rho(J, I) + rho(J - 1, I))/2)*dx*dx/ahv_p(J - 1, I)

def ap_p(J, I):
    return ap_w(J, I) + ap_e(J, I) + ap_s(J, I) + ap_n(J, I)

# 벽 경계조건 소스항 고려 압력 계산 계수
def ap_ww(J, I):
    return ((rho(J, I) + rho(J, I - 1))/2)*dy*dy/(ahu_p(J, I - 1) - Sp_wall_u)

def ap_ew(J, I):
    return ((rho(J, I) + rho(J, I + 1))/2)*dy*dy/(ahu_p(J, I) - Sp_wall_u)

def ap_sw(J, I):
    return ((rho(J, I) + rho(J + 1, I))/2)*dx*dx/(ahv_p(J, I) - Sp_wall_v)

def ap_nw(J, I):
    return ((rho(J, I) + rho(J - 1, I))/2)*dx*dx/(ahv_p(J - 1, I) - Sp_wall_v)

#수정 압력 계수
def apr_w(J, I):
    return ap_w(J, I) * alpha

def apr_e(J, I):
    return ap_e(J, I) * alpha

def apr_s(J, I):
    return ap_s(J, I) * alpha

def apr_n(J, I):
    return ap_n(J, I) * alpha

def bpr(J, I):
    return ((rho(J, I) + rho(J, I - 1)) / 2 * u0[J, I - 1] - (rho(J, I) + rho(J, I + 1)) / 2 * u0[J, I]) * dy \
        + ((rho(J, I) + rho(J + 1, I)) / 2 * v0[J, I] - (rho(J, I) + rho(J - 1, I)) / 2 * v0[J - 1, I]) * dx

def apr_p(J, I):
    return apr_w(J, I) + apr_e(J, I) + apr_s(J, I) + apr_n(J, I)

# 벽 경계조건 소스항 고려 수정압력 계수
def apr_ww(J, I):
    return ap_ww(J, I) * alpha

def apr_ew(J, I):
    return ap_ew(J, I) * alpha

def apr_sw(J, I):
    return ap_sw(J, I) * alpha

def apr_nw(J, I):
    return ap_nw(J, I) * alpha

def c_u1():

    # 압력/속도
    global p
    global u
    global v

    # 추정 압력/속도
    global p0
    global u0
    global v0

    # 수정 압력/속도
    global p1
    global u1
    global v1

    nu = n - 1
    um = m

    ut = np.zeros((um * nu, um * nu))
    uc = np.zeros(um * nu)

    h = 0

    #외부 격자
    for i in range(1, nu - 1):
        ut[h, i] = 1
        h = h + 1
        ut[h, nu * (um - 1) + i] = 1
        h = h + 1

    for j in range(0, um):
        #inlet
        ut[h, nu * j] = 1
        uc[h] = u_inlet
        h = h + 1
        ut[h, nu * j + nu - 1] = 1
        h = h + 1

    #내부 격자
    for i in range(1, nu - 1):
        for j in range(2, um - 2):
            ut[h, nu * j + i] = ahu_p(j, i)/alpha
            ut[h, nu * j + i + 1] = -ahu_e(j, i)
            ut[h, nu * j + i - 1] = -ahu_w(j, i)
            ut[h, nu * (j + 1) + i] = -ahu_s(j, i)
            ut[h, nu * (j - 1) + i] = -ahu_n(j, i)
            uc[h] = (p0[j, i] - p0[j, i + 1])*dy + (1-alpha)*ahu_p(j, i)/alpha * u0[j, i]
            h = h + 1

    #wall
    for i in range(1, nu - 1):
        # top wall
        j = 1
        ut[h, nu * j + i] = (ahu_p(j, i) - Sp_wall_u)/alpha
        ut[h, nu * j + i + 1] = -ahu_e(j, i)
        ut[h, nu * j + i - 1] = -ahu_w(j, i)
        ut[h, nu * (j + 1) + i] = -ahu_s(j, i)
        ut[h, nu * (j - 1) + i] = -ahu_n(j, i)
        uc[h] = (p0[j, i] - p0[j, i + 1])*dy + (1-alpha)*(ahu_p(j, i) - Sp_wall_u)/alpha*u0[j, i]
        h = h + 1
        # bottom wall
        j = um - 2
        ut[h, nu * j + i] = (ahu_p(j, i) - Sp_wall_u)/alpha
        ut[h, nu * j + i + 1] = -ahu_e(j, i)
        ut[h, nu * j + i - 1] = -ahu_w(j, i)
        ut[h, nu * (j + 1) + i] = -ahu_s(j, i)
        ut[h, nu * (j - 1) + i] = -ahu_n(j, i)
        uc[h] = (p0[j, i] - p0[j, i + 1])*dy + (1-alpha)*(ahu_p(j, i) - Sp_wall_u)/alpha*u0[j, i]
        h = h + 1

    u1_flat = np.linalg.solve(ut, uc)
    u1 = u1_flat.reshape(um, nu)


def c_v1():

    # 압력/속도
    global p
    global u
    global v

    # 추정 압력/속도
    global p0
    global u0
    global v0

    # 수정 압력/속도
    global p1
    global u1
    global v1

    nv = n
    mv = m - 1

    vt = np.zeros((mv * nv, mv * nv))
    vc = np.zeros(mv * nv)

    h = 0

    #외부 격자
    for i in range(0, nv):
        vt[h, i] = 1
        vc[h] = v_top
        h = h + 1
        vt[h, nv * (mv - 1) + i] = 1
        vc[h] = v_bottom
        h = h + 1

    for j in range(1, mv - 1):
        #inlet
        vt[h, nv * j] = 1
        h = h + 1
        vt[h, nv * j + nv - 1] = 1
        h = h + 1

    #내부 격자
    for i in range(1, nv - 1):
        for j in range(1, mv - 1):
            vt[h, nv * j + i] = ahv_p(j, i)/alpha
            vt[h, nv * j + i + 1] = -ahv_e(j, i)
            vt[h, nv * j + i - 1] = -ahv_w(j, i)
            vt[h, nv * (j + 1) + i] = -ahv_s(j, i)
            vt[h, nv * (j - 1) + i] = -ahv_n(j, i)
            vc[h] = (p0[j + 1, i] - p0[j, i])*dx + (1-alpha)*ahv_p(j, i)/alpha*v0[j, i]
            h = h + 1

    v1_flat = np.linalg.solve(vt, vc)
    v1 = v1_flat.reshape(mv, nv)

def c_p1():

    # 압력/속도
    global p
    global u
    global v

    # 추정 압력/속도
    global p0
    global u0
    global v0

    # 수정 압력/속도
    global p1
    global u1
    global v1

    mt = np.zeros((m * n, m * n))
    mc = np.zeros(m * n)

    h = 0  # 수정 압력 계산용 행렬 인덱스

    # 계산 영역 외부 절점
    for i in range(0, n):
        mt[h, i] = 1
        h = h + 1
        mt[h, n * (m - 1) + i] = 1
        h = h + 1

    for j in range(1, m - 1):
        mt[h, n * j] = 1
        h = h + 1
        mt[h, n * j + n - 1] = 1
        h = h + 1

    # 계산 영역(1~m-2, 1~n-2)
    # 내부 절점(2~m-3, 2~n-3)
    for i in range(2, n - 2):
        for j in range(2, m - 2):
            mt[h, n * j + i] = apr_p(j, i)
            mt[h, n * j + i + 1] = -apr_e(j, i)
            mt[h, n * j + i - 1] = -apr_w(j, i)
            mt[h, n * (j + 1) + i] = -apr_s(j, i)
            mt[h, n * (j - 1) + i] = -apr_n(j, i)
            mc[h] = bpr(j, i)
            h = h + 1

    # inlet (2~m-3, 1)
    for j in range(2, m - 2):
        mt[h, n * j + 1] = apr_e(j, 1) + apr_s(j, 1) + apr_n(j, 1)
        mt[h, n * j + 1 + 1] = -apr_e(j, 1)
        mt[h, n * (j + 1) + 1] = -apr_s(j, 1)
        mt[h, n * (j - 1) + 1] = -apr_n(j, 1)
        mc[h] = bpr(j, 1)
        h = h + 1

    # outlet (2~m-3, n-2)
    for j in range(1, m - 1):
        mt[h, n * j + n - 2] = 1
        mc[h] = 0
        h = h + 1

    # wall
    # top(1, 2~n-3)
    for i in range(2, n - 2):
        mt[h, n + i] = apr_ww(1, i) + apr_ew(1, i) + apr_s(1, i)
        mt[h, n + i + 1] = -apr_ew(1, i)
        mt[h, n + i - 1] = -apr_ww(1, i)
        mt[h, n + i + n] = -apr_s(1, i)
        mc[h] = bpr(1, i)
        h = h + 1

    # (1, 1)
    mt[h, n + 1] = apr_ew(1, 1) + apr_s(1, 1)
    mt[h, n + 1 + 1] = -apr_ew(1, 1)
    mt[h, n + 1 + n] = -apr_s(1, 1)
    mc[h] = bpr(1, 1)
    h = h + 1

    # bottom(m-2, 2~n-3)
    for i in range(2, n - 2):
        mt[h, n * (m - 2) + i] = apr_ww(m - 2, i) + apr_ew(m - 2, i) + apr_n(m - 2, i)
        mt[h, n * (m - 2) + i + 1] = -apr_ew(m - 2, i)
        mt[h, n * (m - 2) + i - 1] = -apr_ww(m - 2, i)
        mt[h, n * (m - 2) + i - n] = -apr_n(m - 2, i)
        mc[h] = bpr(m - 2, i)
        h = h + 1

    # (m - 2, 1)
    mt[h, n * (m - 2) + 1] = apr_ew(m - 2, 0) + apr_n(m - 2, 1)
    mt[h, n * (m - 2) + 1 + 1] = -apr_ew(m - 2, 1)
    mt[h, n * (m - 2) + 1 - n] = -apr_n(m - 2, 1)
    mc[h] = bpr(m - 2, 1)
    h = h + 1

    p1_flat = np.linalg.solve(mt, mc)
    p1 = p1_flat.reshape(m, n)


def main():

    # 압력/속도
    global p
    global u
    global v

    # 추정 압력/속도
    global p0
    global u0
    global v0

    # 수정 압력/속도
    global p1
    global u1
    global v1

    #잔차
    global Ures
    global Vres

    initial_condition()
    boundary_condition()

    # 해석 시작
    for k in range(r):

        # 이산화 운동량 방정식 풀이
        c_u1()
        c_v1()

        #수정 압력 계산
        c_p1()

        # 보정
        #p 보정
        # 압력 및 속도 업데이트
        for i in range(1, n - 1):
            for j in range(1, m):
                p[j, i] = p1[j, i] * alpha + p0[j, i]
        # u보정
        for i in range(1, n - 2):
            for j in range(2, m - 2):
                u[j, i] = u1[j, i] + (p1[j, i] - p1[j, i + 1]) * dy * alpha / ahu_p(j, i)
        # wall 소스항 포함
        for i in range(1, n - 2):
            u[1, i] = u1[1, i] + (p1[1, i] - p1[1, i + 1]) * dy * alpha / (ahu_p(1, i) - Sp_wall_u)
            u[m - 2, i] = u1[m - 2, i] + (p1[m - 2, i] - p1[m - 2, i + 1]) * dy * alpha \
                          / (ahu_p(m - 2, i) - Sp_wall_u)
        #inlet outlet 설정
        u[1:-1, 0] = u_inlet
        u[1:-1, n - 2] = u[1:-1, n - 3]

        # v보정
        for i in range(1, n - 1):
            for j in range(1, m - 2):
                v[j, i] = v1[j, i] + (p1[j + 1, i] - p1[j, i]) * dx * alpha / ahv_p(j, i)
        #inlet outlet 설정
        v[1:-1, 0] = 0
        v[1:-1, n - 1] = v[1:-1, n - 2]

        # 잔차 계산
        # ures
        ures = 0
        # 내부 격자
        for i in range(1, n - 1 - 1):
            for j in range(2, m - 2):
                ures = ures + abs(ahu_p(j, i) / alpha * u[j, i] - ahu_e(j, i) * u[j, i + 1] \
                       - ahu_w(j, i) * u[j, i - 1] - ahu_s(j, i) * u[j + 1, i] - ahu_n(j, i) * u[j - 1, i] \
                       - ((p0[j, i] - p0[j, i + 1]) * dy + (1 - alpha) * ahu_p(j, i) / alpha * u0[j, i]))

        # wall
        for i in range(1, n - 1 - 1):
            # top wall
            j = 1
            ures = ures + abs(ahu_p(j, i) / alpha * u[j, i] - ahu_e(j, i) * u[j, i + 1] \
                   - ahu_w(j, i) * u[j, i - 1] - ahu_s(j, i) * u[j + 1, i] - ahu_n(j, i) * u[j - 1, i] \
                   - ((p0[j, i] - p0[j, i + 1]) * dy + (1 - alpha) * (ahu_p(j, i) - Sp_wall_u) / alpha * u0[j, i]))
            # bottom wall
            j = m - 2
            ures = ures + abs(ahu_p(j, i) / alpha * u[j, i] - ahu_e(j, i) * u[j, i + 1] \
                   - ahu_w(j, i) * u[j, i - 1] - ahu_s(j, i) * u[j + 1, i] - ahu_n(j, i) * u[j - 1, i] \
                   - ((p0[j, i] - p0[j, i + 1]) * dy + (1 - alpha) * (ahu_p(j, i) - Sp_wall_u) / alpha * u0[j, i]))

        #vres
        vres = 0
        for i in range(1, n - 1):
            for j in range(1, m - 1 - 1):
                vres = vres + abs(ahv_p(j, i) / alpha * v[j, i] -ahv_e(j, i) * v[j, i + 1]
                                      -ahv_w(j, i) * v[j, i - 1] -ahv_s(j, i) * v[j + 1, i] -ahv_n(j, i) * v[j - 1, i]
                                      -((p0[j + 1, i] - p0[j, i]) * dx + (1 - alpha) * ahv_p(j, i) / alpha * v0[j, i]))

        # 업데이트
        u0[:, :] = u[:, :]
        v0[:, :] = v[:, :]
        p0[:, :] = p[:, :]

        print((k + 1) / r * 100, '%')
        print('ures:', ures)
        print('vres:', vres)
        print('')


    # 시각화
    u_nodes = 0.5 * (u[1:-1, :-1] + u[1:-1, 1:])  # Average in the i-direction
    v_nodes = 0.5 * (v[:-1, 1:-1] + v[1:, 1:-1])  # Average in the j-direction

    # Calculate the speed (magnitude of velocity)
    speed = np.sqrt(u_nodes ** 2 + v_nodes ** 2)

    # Create meshgrid for plotting within the specified range (1 <= i <= m-2, 1 <= j <= n-2)
    x = np.linspace(0, 1, n)[1:-1]
    y = np.linspace(0, 1, m)[1:-1]
    X, Y = np.meshgrid(x, y)

    # 압력 필드 시각화
    plt.figure(figsize=(10, 5))
    cp = plt.contourf(X, Y, p0[1:-1, 1:-1], 50)  # 50은 등 고선 수
    plt.colorbar(cp)
    plt.title('Pressure field_SIMPLE/Hybrid')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    plt.figure(figsize=(10, 5))
    speed_plot = plt.pcolormesh(X, Y, speed, shading='gouraud')
    plt.colorbar(speed_plot)
    plt.title('Velocity Magnitude_SIMPLE/Hybrid')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    plt.figure(figsize=(10, 5))
    magnitude = np.sqrt(u_nodes ** 2 + v_nodes ** 2)
    quiver = plt.quiver(X, Y, u_nodes, v_nodes, magnitude, angles='xy', scale_units='xy', scale=100, cmap='jet',
                        width=0.002)
    plt.colorbar(quiver, label='Velocity Magnitude')
    plt.title('Velocity Vector Field_SIMPLE/Hybrid')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == '__main__':
    main()