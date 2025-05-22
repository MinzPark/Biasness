import streamlit as st
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, norm

st.set_page_config(layout="wide")
# … (사이드바 CSS 생략) …

st.title("The Example of Biased Estimator under Poisson Regression")

# 1) 데이터 생성
np.random.seed(0)
N = 100
x = np.random.uniform(0, 10, N)
beta0, beta1 = 0.5, 0.3
mu_true = np.exp(beta0 + beta1 * x)
y = np.random.poisson(mu_true)
x_line = np.linspace(x.min(), x.max(), 100)

# 2) 슬라이더 → n 계산 (지오메트릭 맵)
bias = st.sidebar.slider("Bias (0 = 없음, 0.1 = 매우 큼)", 0.0, 0.1, 0.1, 0.01)
min_n, max_n = 5, 100
t = 1 - (bias / 0.1)
n = int(min_n * (max_n/min_n)**t)
st.sidebar.markdown(f"**샘플 수 n = {n}**")

# 3) Static 부분 캐시 함수
@st.cache_data(show_spinner=False)
def build_base_fig(x, y, beta0, beta1, x_line):
    fig = go.Figure()
    # 관측점
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=np.zeros_like(y),
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.3),
        name='Observations'
    ))
    # ridgeline
    n_ridges = 4
    x_ridg = np.linspace(x.min(), x.max(), n_ridges)
    dx = (x.max()-x.min())/n_ridges
    for xi in x_ridg:
        mu_s = np.exp(beta0 + beta1*xi)
        np.random.seed(42)
        y_samp = np.random.normal(mu_s, mu_s*0.7, 10000)
        kde = gaussian_kde(y_samp)
        yg = np.linspace(y_samp.min(), y_samp.max(), 200)
        dens = kde(yg)/kde(yg).max()*dx*0.8
        Xp = np.full((2, yg.size), xi)
        Yp = np.vstack([yg, yg])
        Zp = np.vstack([np.zeros_like(dens), dens])
        fig.add_trace(go.Surface(
            x=Xp, y=Yp, z=Zp,
            surfacecolor=Zp, colorscale='Blues',
            showscale=False, opacity=0.5, showlegend=False
        ))
        idxm = np.argmax(dens)
        fig.add_trace(go.Scatter3d(
            x=[xi], y=[yg[idxm]], z=[dens[idxm]],
            mode='markers', marker=dict(size=4, color='blue'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[xi, xi], y=[yg[idxm], yg[idxm]], z=[0, dens[idxm]],
            mode='lines', line=dict(color='blue', width=5),
            showlegend=False
        ))
    # unbiased density surface
    y_min, y_max = y.min(), y.max()
    yg2 = np.linspace(y_min, y_max, 100)
    Xg, Yg = np.meshgrid(x_line, yg2)
    Mu = np.tile(np.exp(beta0+beta1*x_line),(yg2.size,1))
    Sigma = np.sqrt(Mu)
    pdf = norm.pdf(Yg, loc=Mu, scale=Sigma)
    scale2 = mu_true.max()*0.3
    fig.add_trace(go.Surface(
        x=Xg, y=Yg, z=pdf*scale2,
        colorscale='Viridis', opacity=0.6, showscale=False,
        name='Smooth density'
    ))
    # layout (camera 고정)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='x'),
            yaxis=dict(title='y'),
            zaxis=dict(title='PDF of y', range=[0, scale2*1.1]),
            aspectratio=dict(x=1,y=1,z=0.5),
            camera=dict(eye=dict(x=1.5,y=1.2,z=0.8))
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        width=900, height=900
    )
    return fig

# 4) 캐시된 Figure 복제 & GLM 회귀선만 추가
base_fig = build_base_fig(x, y, beta0, beta1, x_line)
fig1 = go.Figure(base_fig)

idx = np.random.choice(len(x), size=n, replace=True)
X_s = sm.add_constant(x[idx])
y_s = y[idx]
res = sm.GLM(y_s, X_s, family=sm.families.Poisson()).fit()
mu_hat = res.predict(sm.add_constant(x_line))

fig1.add_trace(go.Scatter3d(
    x=x_line, y=mu_hat, z=np.zeros_like(x_line),
    mode='lines', line=dict(color='orange', width=4),
    name=f'GLM μ(x) (n={n})'
))

# 5) 두 그래프 나란히
col1, col2 = st.columns(2)
with col1:
    st.markdown("<h3 style='text-align:center'>Biased Estimator vs Real Density of target</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.markdown("<h3 style='text-align:center'>Unbiased Estimator</h3>", unsafe_allow_html=True)
    st.plotly_chart(base_fig, use_container_width=True)
