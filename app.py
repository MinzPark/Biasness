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
    /* 사이드바 전체 너비 */
    [data-testid="stSidebar"] { min-width: 100px !important; }
    [data-testid="stSidebarNav"] > div:first-child { width: 100px !important; }
    /* 사이드바 내용 폰트 크기 조절 */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        font-size: 20px !important;
    }
    @media only screen and (max-width: 480px) {
      .block-container {
        padding: 0.5rem !important;
      }
      .stPlotlyChart > div {
        height: 50vh !important;
      }
    }
    </style>
    ''', unsafe_allow_html=True
)


st.title("The Example of Biased Estimator under Poisson Regression")
st.sidebar.markdown("<h2 style='font-size:30px; margin-bottom: 1rem;'>Parameters</h2>", unsafe_allow_html=True)


# ---------------------



# # 1) 데이터 생성 & Poisson GLM fitting
# np.random.seed(int(0))
# N = 100
# x = np.random.uniform(0, 10, N)
# beta0, beta1 = 0.5, 0.3
# mu_true = np.exp(beta0 + beta1 * x)
# y = np.random.poisson(mu_true)
# # 2) 기본 회귀선 x축
# x_line = np.linspace(x.min(), x.max(), 100)




# fig1 = go.Figure()

# # 3.1 관측점
# fig1.add_trace(go.Scatter3d(
#     x=x, y=y, z=np.zeros_like(y),
#     mode='markers',
#     marker=dict(size=2, color='blue', opacity=0.3),
#     name='Observations'
# ))

# # 3.2 KDE ridgeline 시각화
# n_ridges = 4
# x_ridg = np.linspace(x.min(), x.max(), n_ridges)

# for xi in x_ridg:
#     mu_sample = np.exp(beta0 + beta1 * xi)
#     y_sample = np.random.normal(loc=mu_sample, scale=mu_sample * 0.7, size=10000)

#     kde_sample = gaussian_kde(y_sample)
#     y_sample_grid = np.linspace(y_sample.min(), y_sample.max(), 200)
#     dens_sample = kde_sample(y_sample_grid)

#     Xp_sample = np.full((2, y_sample_grid.size), xi)
#     Yp_sample = np.vstack([y_sample_grid, y_sample_grid])
#     Zp_sample = np.vstack([np.zeros_like(dens_sample), dens_sample])

#     fig1.add_trace(go.Surface(
#         x=Xp_sample,
#         y=Yp_sample,
#         z=Zp_sample,
#         surfacecolor=Zp_sample,
#         colorscale='Blues',
#         cmin=0,
#         cmax=dens_sample.max(),
#         showscale=False,
#         opacity=0.5,
#         showlegend=False
#     ))

#     # peak 표시
#     idx_max = np.argmax(dens_sample)
#     y_peak = y_sample_grid[idx_max]
#     z_peak = dens_sample[idx_max]

#     fig1.add_trace(go.Scatter3d(
#         x=[xi], y=[y_peak], z=[z_peak],
#         mode='markers', marker=dict(size=2, color='blue'),
#         showlegend=False
#     ))

#     fig1.add_trace(go.Scatter3d(
#         x=[xi, xi], y=[y_peak, y_peak], z=[0, z_peak],
#         mode='lines',
#         line=dict(color='blue', width=3, dash='solid'),
#         showlegend=False
#     ))


# # ---------------------
# # 슬라이더: bias 조절 (0 = 무편향, 1 = 높은 편향)
# # ---------------------
# # 1) 파라미터
# bias = st.sidebar.slider("Bias (0 = 없음, 0.5 = 매우 큼)", 0.0, 0.1, 0.1, 0.01)
# min_n, max_n = 5, 100

# t = 1- (bias / 0.1)
# n = int( min_n * (max_n/min_n)**t )

# # ---------------------
# # 4) GLM 모델 피팅 & 회귀선 시각화
# # ---------------------
# idx = np.random.choice(len(x), size=n, replace=True)
# X_sample = sm.add_constant(x[idx])
# y_sample = y[idx]
# res = sm.GLM(y_sample, X_sample, family=sm.families.Poisson()).fit()
# mu_line = res.predict(sm.add_constant(x_line))

# fig1.add_trace(go.Scatter3d(
#     x=x_line, y=mu_line, z=np.zeros_like(x_line),
#     mode='lines',
#     line=dict(color='orange', width=4),
#     name='GLM μ(x)'
# ))

# # ---------------------
# # 5) 정답 회귀선도 같이 그림
# # ---------------------
# mu_true_line = np.exp(beta0 + beta1 * x_line)
# fig1.add_trace(go.Scatter3d(
#     x=x_line, y=mu_true_line, z=np.zeros_like(x_line),
#     mode='lines',
#     line=dict(color='red', width=4, dash='dot')
# ))

np.random.seed(0)
N = 100
x = np.random.uniform(0, 10, N)
beta0, beta1 = 0.5, 0.3
mu_true = np.exp(beta0 + beta1 * x)
y = np.random.poisson(mu_true)
x_line = np.linspace(x.min(), x.max(), 100)

# 2) 슬라이더 → n 계산
bias = st.sidebar.slider("Bias (0 = 없음, 0.1 = 매우 큼)", 0.0, 0.1, 0.1, 0.01)
min_n, max_n = 5, 100
t = 1 - (bias / 0.1)
n = int(min_n * (max_n / min_n) ** t)
st.sidebar.markdown(f"**샘플 수 n = {n}**")

# 3) ridgeline + 관측점 한번만 그리는 함수 (캐시)
@st.cache_data
def build_base_fig(x, y, beta0, beta1):
    fig = go.Figure()
    # Observations
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=np.zeros_like(y),
        mode='markers',
        marker=dict(size=2, color='blue', opacity=0.3),
        name='Observations'
    ))

    # KDE ridgeline
    n_ridges = 4
    x_ridg = np.linspace(x.min(), x.max(), n_ridges)
    dx = (x.max() - x.min()) / n_ridges
    for xi in x_ridg:
        mu_s = np.exp(beta0 + beta1 * xi)
        # (원하시면 샘플 수나 grid 개수를 줄이세요)
        y_samp = np.random.normal(mu_s, mu_s * 0.7, 5000)
        kde = gaussian_kde(y_samp)
        yg = np.linspace(y_samp.min(), y_samp.max(), 100)
        dens = kde(yg)
        dens = dens / dens.max() * dx * 0.8

        Xp = np.full((2, yg.size), xi)
        Yp = np.vstack([yg, yg])
        Zp = np.vstack([np.zeros_like(dens), dens])

        fig.add_trace(go.Surface(
            x=Xp, y=Yp, z=Zp,
            surfacecolor=Zp, colorscale='Blues',
            showscale=False, opacity=0.5,
            showlegend=False
        ))

        idxm = np.argmax(dens)
        fig.add_trace(go.Scatter3d(
            x=[xi], y=[yg[idxm]], z=[dens[idxm]],
            mode='markers',
            marker=dict(size=2, color='blue'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[xi, xi], y=[yg[idxm], yg[idxm]], z=[0, dens[idxm]],
            mode='lines',
            line=dict(color='blue', width=3),
            showlegend=False
        ))

    # camera 고정
    fig.update_layout(scene=dict(camera=dict(eye=dict(x=1.5, y=1.2, z=0.8))))
    return fig

base_fig = build_base_fig(x, y, beta0, beta1)

# 4) Base를 복제하고 오렌지 GLM 선만 추가
fig1 = go.Figure(base_fig)

# Poisson GLM 피팅
idx = np.random.choice(len(x), size=n, replace=True)
res = sm.GLM(y[idx], sm.add_constant(x[idx]), family=sm.families.Poisson()).fit()
mu_hat = res.predict(sm.add_constant(x_line))

fig1.add_trace(go.Scatter3d(
    x=x_line, y=mu_hat, z=np.zeros_like(x_line),
    mode='lines',
    line=dict(color='orange', width=4),
    name='GLM μ(x)'
))

# 5) true μ(x) 선은 범례에서 숨기기
mu_true_line = np.exp(beta0 + beta1 * x_line)
fig1.add_trace(go.Scatter3d(
    x=x_line, y=mu_true_line, z=np.zeros_like(x_line),
    mode='lines',
    line=dict(color='red', width=4, dash='dot'),
    showlegend=False
))

fig1.update_layout(
    scene=dict(
        xaxis=dict(title='x', range=[x.min(), x.max()]),
        yaxis=dict(title='y', range=[y.min()-3, y.max()+10]),
        zaxis=dict(title='Probability Density Function of Y', range=[0, 0.5]),
        aspectratio=dict(x=1, y=1, z=0.5),
        camera=dict(eye=dict(x=1.5, y=1.2, z=0.8))
    ),
    margin=dict(l=0, r=0, t=0, b=0), width=600, height=600,
    legend=dict(
            x=1,           # paper 좌우 기준(0~1)에서 우측 95%
            y=1,           # paper 상하 기준(0~1)에서 위쪽 95%
            xanchor='right',  # x=0.95 지점을 legend 우측에 맞춤
            yanchor='top',    # y=0.95 지점을 legend 상단에 맞춤
            bgcolor='rgba(255,255,255,0.5)',  # 배경 반투명
            borderwidth=0,
            font=dict(size=10),  # legend 텍스트 크기
            itemsizing='constant',  # itemwidth를 고정 사이즈로 사용
            itemwidth=50            # 각 legend item 너비(px)
        ),
)


fig1.data[-1].showlegend = False


# ──────────────────────────────────────────────────────
# Figure 2: Unbiased Estimato
y_min, y_max = y.min(), y.max()
y_grid2 = np.linspace(y_min, y_max, 100)
Xg, Yg = np.meshgrid(x_line, y_grid2)
Mu = np.tile(mu_line, (y_grid2.size, 1))
Sigma = np.sqrt(Mu)
pdf = norm.pdf(Yg, loc=Mu, scale=Sigma)
# scale2 = mu_line.max() * 0.3

fig2 = go.Figure()
fig2.add_trace(go.Scatter3d(
    x=x_line, y=mu_line, z=np.zeros_like(x_line),
    mode='lines', line=dict(color='orange', width=4),
    name='Regression μ(x)'
))
fig2.add_trace(go.Surface(
    x=Xg, y=Yg, z=pdf,
    colorscale='Viridis', opacity=0.6, showscale=False,
    name='Smooth density'
))
fig2.update_layout(
    scene=dict(
        xaxis=dict(range=[x.min(), x.max()]),
        yaxis=dict(range=[y_min, y_max]),
        zaxis=dict(title = 'Probability Density Function of y',range=[0, 0.5]),
        aspectratio=dict(x=1, y=1, z=0.5)
    ),
    margin=dict(l=0, r=0, t=0, b=0), 
    width=600, height=600
)

# ──────────────────────────────────────────────────────
# 두 개의 Figure를 가로로 나란히 배치
col1, col2 = st.columns(2)
with col1:
    st.markdown("<h3 style='text-align:center'>Biased Estimator vs Real Density of target</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig1, use_container_width=True, config={"responsive": True})
with col2:
    st.markdown("<h3 style='text-align:center'>Unbiased Estimator</h3>", unsafe_allow_html=True)
    # st.subheader("Unbiased Estimator")
    st.plotly_chart(fig2, use_container_width=True, config={"responsive": True})
