# app.py
import streamlit as st
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, norm

st.set_page_config(layout="wide")
# 사이드바 너비 조정 (px 단위)

st.markdown(
    '''
    <style>
    /* 사이드바 왼쪽 여백 제거 */
    section[data-testid="stSidebar"] {
        padding-left: 0rem !important;
        margin-left: 0rem !important;
    }
    
    /* 사이드바 내부 내용 패딩 제거 */
    section[data-testid="stSidebar"] > div:first-child {
        padding: 0rem !important;
        min-width: 200px !important; max-width: 200px !important;
    }

    /* 본문 block-container의 왼쪽 패딩 제거 → 사이드바와 붙음 */
    div.block-container {
        padding-left: 0.5rem !important;  /* 최소한의 여백만 남기기 */
        padding-right: 1rem !important;
    }
    /* 사이드바 전체 너비 */
    /*[data-testid="stSidebar"] { min-width: 200px !important; max-width: 200px !important; }*/
    [data-testid="stSidebarNav"] > div:first-child { width: 200px !important; }
    /* 사이드바 내용 폰트 크기 조절 */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        font-size: 15px !important;
    }
    </style>
    ''', unsafe_allow_html=True
)


st.title("The Example of Biased Estimator under Poisson Regression")

st.sidebar.markdown("<h2 style='font-size:30px; margin-bottom: 1rem;'>Parameters</h2>", unsafe_allow_html=True)

# 사이드바에서 조정 가능한 시드, 샘플 수, 회귀 파라미터
seed = st.sidebar.number_input("Random seed", value=0, step=1)
N = st.sidebar.slider("Number of samples (N)", 50, 1000, 200)
beta0 = st.sidebar.slider("β₀ (intercept)", 0.0, 2.0, 0.5, 0.1)
beta1 = st.sidebar.slider("β₁ (slope)", 0.0, 1.0, 0.3, 0.05)


# 1) 데이터 생성 & Poisson GLM fitting
np.random.seed(int(seed))
x = np.random.uniform(0, 10, N)
mu_true = np.exp(beta0 + beta1 * x)
y = np.random.poisson(mu_true)

X = sm.add_constant(x)
res = sm.GLM(y, X, family=sm.families.Poisson()).fit()

# 2) 회귀선 계산
x_line = np.linspace(x.min(), x.max(), 100)
mu_line = res.predict(sm.add_constant(x_line))

# ──────────────────────────────────────────────────────
# Figure 1: Biased Estimator vs Real Density of target
n_ridges = 4
x_ridg = np.linspace(x.min(), x.max(), n_ridges)
dx = (x.max() - x.min()) / n_ridges
y_grid = np.linspace(y.min(), y.max(), 200)
scale = dx * 0.8

fig1 = go.Figure()

# Observations
fig1.add_trace(go.Scatter3d(
    x=x, y=y, z=np.zeros_like(y),
    mode='markers',
    marker=dict(size=3, color='blue', opacity=0.3),
    name='Observations'
))

# GLM line
fig1.add_trace(go.Scatter3d(
    x=x_line, y=mu_line, z=np.zeros_like(x_line),
    mode='lines', line=dict(color='orange', width=4),
    name='GLM μ(x)'
))

# 4.3 각 x_ridg 위치에서의 local KDE ridgeline
for xi in x_ridg:
    mask = (x >= xi-dx/2) & (x < xi+dx/2)
    y_sub = y[mask]
    if len(y_sub) < 5:
        continue
    kde = gaussian_kde(y_sub)
    dens = kde(y_grid)
    dens = dens / dens.max() * scale


    # surface 아래쪽(z=0)부터 위쪽(z=dens)까지 면으로 채우기
    Xp = np.full((2, y_grid.size), xi)       # shape (2, N)
    Yp = np.vstack([y_grid, y_grid])         # shape (2, N)
    Zp = np.vstack([np.zeros_like(dens),      # 아래 면 (z=0)
                    dens])                   # 위 면 (z=dens)


    fig1.add_trace(go.Surface(
        x=Xp, y=Yp, z=Zp,
        surfacecolor=Zp,            # 면 내부 색상도 dens 기반
        colorscale='Blues',
        cmin=0, cmax=dens.max(),
        showscale=False,
        opacity=0.5,
        showlegend=False
    ))

# 4.4 각 ridgeline 위치에서 모델 μ(x)와 관측 μ_emp 간 편향(bias) 표시
for xi in x_ridg:
    # 1) 해당 x 구간 데이터 추출
    mask = (x >= xi-dx/2) & (x < xi+dx/2)
    y_sub = y[mask]
    if len(y_sub) < 5:
        continue

    # 2) KDE 밀도 계산 및 스케일링
    dens = gaussian_kde(y_sub)(y_grid)
    dens = dens / dens.max() * scale

    # 3) 면(surface) 그리기
    Xp = np.full((2, y_grid.size), xi)
    Yp = np.vstack([y_grid, y_grid])
    Zp = np.vstack([np.zeros_like(dens), dens])
    fig1.add_trace(go.Surface(
        x=Xp, y=Yp, z=Zp,
        surfacecolor=Zp, colorscale='Blues',
        opacity=0.3,
        showscale=False, showlegend=False
    ))

    # 4) 최고점(peak) 좌표 계산
    idx_max = np.argmax(dens)
    y_peak = y_grid[idx_max]
    z_peak = dens[idx_max]

    # 5) peak 위치에 마커
    fig1.add_trace(go.Scatter3d(
        x=[xi], y=[y_peak], z=[z_peak],
        mode='markers', marker=dict(size=3, color='blue'),
        showlegend=False
    ))

    # 6) **여기서 바로** peak 높이선을 추가
    fig1.add_trace(go.Scatter3d(
        x=[xi, xi],            # x 고정
        y=[y_peak, y_peak],    # y=peak 위치로 수직선
        z=[0, z_peak],         # z=0 → z=peak 높이
        mode='lines',
        line=dict(color='blue', width=5, dash='solid'),
        showlegend=False
    ))
fig1.update_layout(
    scene=dict(
        xaxis=dict(title='x', range=[x.min(), x.max()]),
        yaxis=dict(title='y', range=[y.min(), y.max()]),
        zaxis=dict(title='Probability Density Function of Y', range=[0, scale*2]),
        aspectratio=dict(x=1, y=1, z=0.5),
        camera=dict(eye=dict(x=1.5, y=1.2, z=0.8))
    ),
    margin=dict(l=0, r=0, t=0, b=0), width=700, height=600,
    legend=dict(
            x=0.95,           # paper 좌우 기준(0~1)에서 우측 95%
            y=0.9,           # paper 상하 기준(0~1)에서 위쪽 95%
            xanchor='right',  # x=0.95 지점을 legend 우측에 맞춤
            yanchor='top',    # y=0.95 지점을 legend 상단에 맞춤
            bgcolor='rgba(255,255,255,0.5)',  # 배경 반투명
            borderwidth=0,
            font=dict(size=15),  # legend 텍스트 크기
            itemsizing='constant',  # itemwidth를 고정 사이즈로 사용
            itemwidth=60            # 각 legend item 너비(px)
        )
)

# ──────────────────────────────────────────────────────
# Figure 2: Unbiased Estimato
y_min, y_max = y.min(), y.max()
y_grid2 = np.linspace(y_min, y_max, 100)
Xg, Yg = np.meshgrid(x_line, y_grid2)
Mu = np.tile(mu_line, (y_grid2.size, 1))
Sigma = np.sqrt(Mu)
pdf = norm.pdf(Yg, loc=Mu, scale=Sigma)
scale2 = mu_line.max() * 0.3

fig2 = go.Figure()
fig2.add_trace(go.Scatter3d(
    x=x_line, y=mu_line, z=np.zeros_like(x_line),
    mode='lines', line=dict(color='orange', width=4),
    name='Regression μ(x)'
))
fig2.add_trace(go.Surface(
    x=Xg, y=Yg, z=pdf * scale2,
    colorscale='Viridis', opacity=0.6, showscale=False,
    name='Smooth density'
))
fig2.update_layout(
    scene=dict(
        xaxis=dict(range=[x.min(), x.max()]),
        yaxis=dict(range=[y_min, y_max]),
        zaxis=dict(title = 'Probability Density Function of y',range=[0, scale2*1.1]),
        aspectratio=dict(x=1, y=1, z=0.5)
    ),
    margin=dict(l=0, r=0, t=0, b=0), 
    width=700, height=600
)

# ──────────────────────────────────────────────────────
# 두 개의 Figure를 가로로 나란히 배치
col1, col2 = st.columns(2)
with col1:
    st.markdown("<h3 style='text-align:center'>Biased Estimator vs Real Density of target</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.markdown("<h3 style='text-align:center'>Unbiased Estimator</h3>", unsafe_allow_html=True)
    # st.subheader("Unbiased Estimator")
    st.plotly_chart(fig2, use_container_width=True)

