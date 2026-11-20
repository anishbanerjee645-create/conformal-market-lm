from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)


def H(t, l=1):
    doc.add_heading(t, level=l)


def P(t):
    doc.add_paragraph(t)


def EQ(t):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(t)
    r.font.name = "Cambria Math"
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)


def B(t):
    doc.add_paragraph(t, style="List Bullet")


doc.add_heading("Mathematical Model of the NSE Market Predictor", 0)
P(
    "This document describes every mathematical component of the predictor in the Stock_Predictor project: "
    "data and features, the spatio-temporal graph attention network (ST-GATv2), self-supervised "
    "pre-training, adaptive conformal uncertainty, the online self-learning engine, and the evaluation "
    "protocol, and MarketLM, a small GPT-style transformer over market tokens (Section 11). "
    "Note: the project is not a chat-style large language model. The ST-GATv2 is a compact NumPy graph "
    "neural network, and MarketLM is a ~460K-parameter decoder-only transformer. Both are research models, "
    "far smaller than commercial LLMs."
)

H("1. Problem Definition")
P("We observe N = 35 NSE stocks (7 sectors) on trading days t = 1..T (T = 5,834, 2002-2026). "
  "For stock i on day t we have a feature vector x(i,t) in R^F with F = 16. The goal is to predict the next-day "
  "log return and a calibrated prediction interval.")
EQ("r(i,t) = 100 · ln( P(i,t) / P(i,t-1) )          (log return, in %)")
EQ("y(i,t) = r(i,t+1)          (prediction target)")
EQ("Goal: point forecast ŷ(i,t) and interval [L(i,t), U(i,t)] with  P( y ∈ [L,U] ) ≥ 1 − α,  α = 0.10")

H("2. Data and Features")
P("Sources: 26YRS_closes (daily closes, 2000-2026), India_Stock_Market_Data.xlsx (annual GDP growth, CPI "
  "inflation, INR/USD, SENSEX return), and 5-year OHLCV and macro index files. The 16 features per stock per day are:")
for f in [
    "ret_1d, ret_5d, ret_20d: 1/5/20-day log returns",
    "vol_10d, vol_20d: rolling standard deviation of 1-day returns",
    "gk_vol: Garman-Klass range volatility (5-day mean)",
    "vol_zscore: 20-day z-score of log volume",
    "rsi_14 and macd_norm: momentum oscillators",
    "sector_rel_ret: stock return minus its sector mean",
    "nifty_beta_ret: stock return minus market return",
    "india_vix_z: 60-day z-score of India VIX (or realised-volatility proxy before VIX history)",
    "usdinr_ret_5d, brent_ret_5d: currency and crude oil 5-day returns",
    "india_gdp_growth_z, india_cpi_inflation_z: standardised annual macro regime values",
]:
    B(f)
EQ("RSI(t) = [ 100 − 100 / (1 + G(t)/(L(t)+ε)) − 50 ] / 50,    G, L = 14-day mean gain / loss")
EQ("MACD(t) = 100 · [EMA12(P) − EMA26(P)] / P,     macd_norm = clip( MACD − EMA9(MACD), −5, 5 )")
EQ("Garman-Klass:  σ²_GK = 0.5·[ln(H/L)]² − (2 ln 2 − 1)·[ln(C/O)]²")
EQ("Standardisation:  x̃ = clip( (x − μ_F)/σ_F , −4.5, 4.5 )")
P("Limitation: the 26-year file contains closes only, so for the long history open, high, low and "
  "volume are approximated (open = previous close; high/low = ±0.65%; constant volume). Features that "
  "depend on them carry little real information before the 5-year OHLCV window.")

H("3. Multi-Scale Temporal Context")
P("Each stock's standardised features are summarised at three time scales and concatenated (3F = 48 inputs):")
EQ("x_inst(t) = x̃(t)")
EQ("x_short(t) = (1/5) Σ_{k=0..4} x̃(t−k)")
EQ("x_long(t) = (1/L) Σ_{k=0..L−1} x̃(t−k),    L = 15")
EQ("m(i,t) = [ x_inst ‖ x_short ‖ x_long ]  ∈ R^48")

H("4. Dynamic Multi-Relational Contagion Graph")
P("A time-varying edge tensor E(t) in R^{N×N×4} links stocks through four relations:")
EQ("E0 = A_intra:  1 if same sector (row-normalised)")
EQ("E1 = A_supply:  prior cross-sector coupling weight c(s_i, s_j), e.g. Energy-Auto 0.65, Metals-Auto 0.68, Banking-Auto 0.70")
EQ("E2 = A_corr(t):  exponentially weighted rolling correlation (window 45) , thresholded at |ρ| ≥ 0.25, row-normalised")
EQ("ρ_ij(t) = Cov_w(r_i, r_j) / ( σ_w,i · σ_w,j ),     weights w_k ∝ exp(linspace(−1.5, 0, 45))")
EQ("E3 = A_tail(t):  downside lead-lag contagion,  [r⁻_i(t) · r⁻_j(t−1)] / (‖r⁻_i‖ · ‖r⁻_j‖),  r⁻ = min(r, 0)")
P("E2 captures co-movement; E3 captures asymmetric spill-over, i.e. whether a fall in stock j yesterday "
  "tends to be followed by a fall in stock i today.")

H("5. ST-GATv2 Network")
H("5.1 Gated temporal encoder", 2)
EQ("a(i,t) = LeakyReLU( W_temp · m(i,t) + b_temp )")
EQ("g(i,t) = σ( W_gate · m(i,t) + b_gate )")
EQ("h(i,t) = a(i,t) ⊙ g(i,t)       (hidden size D = 32)")
EQ("LeakyReLU(z) = z if z>0 else 0.15 z;     σ(z) = 1/(1+e^{−z})")
H("5.2 Multi-head graph attention (H = 4 heads, Dh = D/H = 8)", 2)
EQ("q_h(i) = LeakyReLU(W_q^h h_i),   k_h(j) = LeakyReLU(W_k^h h_j),   v_h(j) = W_v^h h_j")
EQ("e_ij^h = [ aᵀ q_h(i) + aᵀ k_h(j) + (W_e^h aᵀ)ᵀ · E_ij ] / √Dh")
EQ("α_ij^h = softmax_j( e_ij^h + mask_ij ),     mask_ij = −10⁴ if no edge, else 0")
EQ("z_i^h = Σ_j α_ij^h · v_h(j)")
EQ("z_i = concat_h(z_i^h);   g_i = LeakyReLU( W_out z_i + b_out )")
EQ("u_i = h_i + 0.5 · g_i       (residual graph update)")
P("Attention scores depend on the node states AND the 4-dimensional edge features, so the learned "
  "weights α_ij show which stocks the model treats as most influential for stock i on day t.")
H("5.3 Output heads", 2)
EQ("ŷ_i = W_muᵀ u_i + W_skipᵀ x̃_i + b_mu          (point forecast)")
EQ("s_lo = softplus(W_loᵀ u_i + b_lo) + 0.15,    s_hi = softplus(W_hiᵀ u_i + b_hi) + 0.15")
EQ("q_low = ŷ − s_lo,     q_high = ŷ + s_hi          (conditional quantile band)")
H("5.4 Training loss", 2)
P("The network is trained with a joint robust point loss and quantile (pinball) losses:")
EQ("Huber_δ(e) = ½ e²  if |e| ≤ δ,  else δ(|e| − ½δ),   δ = 1.5,   e = ŷ − y")
EQ("Pinball_τ(y, q) = max( τ (y − q), (τ − 1)(y − q) ),   τ_lo = α/2 = 0.05,  τ_hi = 1 − α/2 = 0.95")
EQ("L = Huber(ŷ, y) + 0.6 · [ Pinball_{0.05}(y, q_low) + Pinball_{0.95}(y, q_high) ]")
H("5.5 Optimiser", 2)
P("AdamW with decoupled weight decay, gradient clipping at ±2, cosine learning-rate decay, "
  "lr = 3e-3, weight decay = 1.5e-3, 45 epochs, batch 64. Gradients are computed by hand-written backpropagation "
  "through the attention softmax:")
EQ("∂L/∂logit_ij = α_ij ( ∂L/∂α_ij − Σ_k α_ik ∂L/∂α_ik ) / √Dh       (softmax backward)")
EQ("m ← β1 m + (1−β1) g;   v ← β2 v + (1−β2) g²;   θ ← θ − lr·[ m̂/(√v̂+ε) + wd·θ ]")

H("6. Self-Supervised Pre-Training (offline)")
P("Before supervised training, the encoder and attention layers are pre-trained on unlabeled windows "
  "with masked feature reconstruction and next-step prediction:")
EQ("M ~ Bernoulli(0.75) element-wise,   X_masked = M ⊙ X̃")
EQ("L_SSL = same joint loss as Section 5.4, input X_masked, target r̃(t+1)        (mask ratio 25%, 2 epochs)")
P("This forces the network to infer a stock's behaviour from its unmasked features and from its graph "
  "neighbours, similar in spirit to masked-token or next-token objectives in language models.")

H("7. Adaptive Conformal Inference (uncertainty)")
P("The raw quantile band is not guaranteed to cover 90% of outcomes, so it is calibrated "
  "online without distributional assumptions.")
EQ("Non-conformity score (CQR):  s_i,t = max( q_low − y , y − q_high )")
EQ("Split conformal:  q̂ = Quantile_{(1−α)(1+1/n)}( s_cal ),   interval = [ q_low − q̂ , q_high + q̂ ]")
EQ("Rolling CQR:  q̂_t = Quantile_{1−α}( s over last 90 days )")
EQ("ACI (Gibbs & Candès 2021):  α_{t+1,i} = clip( α_{t,i} + γ ( α* − err_{t,i} ),  0.005, 0.50 )")
EQ("err_{t,i} = 1 if y_{t,i} ∉ C_{t,i}(α_{t,i}), else 0;     γ = 0.035;   α* = 0.10")
P("When the interval misses too often, α shrinks and the band widens; when it covers too often, the band tightens. "
  "Reported out-of-sample walk-forward empirical coverage: 89.84% against a 90% target.")
H("Interval evaluation", 2)
EQ("Coverage = mean( 1[ L ≤ y ≤ U ] ),      Width = mean( U − L )")

H("8. Online Self-Learning Engine")
P("A second, lightweight learner adapts continuously on live prices. It is a linear model on 7 causal features.")
EQ("φ(t) = [ tanh(r1/2.5), tanh(r5/4.5), tanh(r20/8), RSI_norm, MACD_norm, vol_ratio − 1, 1 ]")
EQ("r1, r5, r20 = % returns over 1/5/20 bars;  RSI_norm = (RSI − 50)/25;  MACD_norm = tanh( (MA5 − MA20)/C · 100/2 )")
EQ("r̂(t+1) = wᵀ φ(t) + b_symbol")
P("Each update draws the newest causal pairs (φ(t), r(t+1)) plus 32 samples from a replay buffer "
  "(to avoid forgetting), randomly masks 20% of the features, and takes one Adam step on the Huber loss:")
EQ("φ_masked = M ⊙ φ,   M ~ Bernoulli(0.80)")
EQ("w ← clip( w − lr · Adam( ∇_w Huber(wᵀ φ_masked − y) + 0.015 w ),  −0.35, 0.35 )")
EQ("b_s ← clip( 0.85 b_s + 0.15 · mean(y − wᵀφ),  −0.25, 0.25 )      (per-symbol bias)")
P("Schedule: every 3 seconds the engine captures live quotes and learns from realised errors. Every 24 hours it "
  "writes the day's open and close to a self-generated dataset (daily_self_learned_dataset.csv) and fine-tunes on "
  "the daily returns.")
P("Bulk pre-training: before going live, this learner was trained with mini-batch Adam on every series in the "
  "26-year file (4,090 tickers, 6.5 million samples; last ~2 years held out).")

H("9. Evaluation Protocol")
for t in [
    "Walk-forward (expanding window) evaluation; the model never sees future data.",
    "Rank IC: Spearman correlation between predicted and realised cross-sectional returns.",
    "Directional accuracy: share of days with sign(ŷ) = sign(y), compared with the base rate.",
    "Diebold-Mariano test of forecast-error differences vs GARCH and ridge baselines.",
    "Long-short portfolio backtest from predicted ranks (Sharpe ratio, drawdown).",
    "Baselines: AR(1)+GARCH(1,1), cross-sectional Ridge, temporal-only network, static-graph network, ST-GAT without tail edges.",
]:
    B(t)
EQ("GARCH(1,1):  σ²_{t+1} = ω + α r_t² + β σ²_t;     AR(1):  μ_t = c + φ r_{t−1}")
EQ("Ridge:  β̂ = (XᵀX + λI)⁻¹ Xᵀ y")
EQ("Rank IC_t = Spearman( ŷ_{·,t}, y_{·,t} )")

H("10. Honest Results and Limitations")
B("Interval calibration works: ACI coverage 89.84% vs 90% target.")
B("MarketLM got 12 minutes of CPU training; its pooled correlation (0.146) is not a within-day cross-sectional rank IC and likely overstates stock-picking skill.")
B("Direction is nearly unpredictable at the 1-day horizon: the bulk-trained online learner scored 50.6% directional accuracy "
  "against a 50.4% base rate on held-out data (rank IC ≈ 0.04). This is consistent with market efficiency.")
B("Closes-only history, survivorship bias (only currently listed stocks), and unadjusted splits limit accuracy.")
B("Forecasts are research estimates, not investment advice.")
H("Suggested improvements", 2)
for t in [
    "Predict volatility-scaled multi-day cross-sectional ranks rather than raw 1-day returns.",
    "Use split-adjusted OHLCV data and add real volume/range features.",
    "Add gradient-boosted trees and a rank-IC loss; ensemble diverse signals.",
    "Gate online updates with a holdout check so live learning does not fit noise.",
]:
    B(t)

H("11. MarketLM: GPT-style Transformer over Market Tokens")
P("MarketLM applies the language-model recipe to prices: each stock-day is one token and the model predicts the next token.")
H("11.1 Tokenisation", 2)
EQ("r_t = 100 · ln(P_t / P_{t−1}),   σ_{t−1} = std(r_{t−20..t−1}),   z_t = clip( r_t / max(σ_{t−1}, 0.3), −6, 6 )")
EQ("token_t = bin_k(z_t),  k = 0..31,  bin edges = 31 quantiles of z on the training period")
P("Only past data enter σ, so tokens are causal. Days with |z| ≥ 6 (typically splits or data errors) are dropped. Each bin has a centre c_k = median of z inside the bin.")
H("11.2 Architecture", 2)
EQ("x_t = E_tok[token_t] + E_pos[t],   t = 1..64      (context 64 days, d = 96)")
EQ("x ← x + MHA( LN(x), causal mask )      (4 heads)")
EQ("x ← x + W2 · GELU( W1 · LN(x) )          (4 blocks, hidden size 4d)")
EQ("p( token_{t+1} | token_{≤t} ) = softmax( W_head · LN(x_t) )")
H("11.3 Objective and training", 2)
EQ("L = −(1/T) Σ_t log p( token_{t+1} | token_{≤t} )        (next-token cross-entropy)")
P("AdamW (lr 1e-3, weight decay 0.05, cosine decay, gradient clip 1.0), batch of 128 windows sampled at random from 3,765 tickers, about 1,800 steps. The last ~2 years are held out.")
H("11.4 From distribution to forecast", 2)
EQ("P(up) = Σ_{k : c_k > 0} p_k,    E[z] = Σ_k p_k c_k,    expected move = E[z] · σ")
EQ("range [5%, 95%] = [ c_{k5}, c_{k95} ] · σ,   k_q = smallest k with Σ_{j≤k} p_j ≥ q")
EQ("Entropy = −Σ_k p_k log2 p_k        (uncertainty score)")
P("Held-out results: next-token loss 3.149 vs 3.333 for a frequency-only baseline; directional accuracy 60.2% vs 59.4% base rate; pooled correlation 0.146. The gain comes mainly from predicting move size (volatility clustering); the directional edge is under one percentage point.")

H("Appendix: Notation")
tbl = doc.add_table(rows=1, cols=2)
tbl.style = "Light Grid Accent 1"
tbl.rows[0].cells[0].text = "Symbol"
tbl.rows[0].cells[1].text = "Meaning"
for a, b in [
    ("N, T, F", "stocks (35), trading days (5,834), features (16)"),
    ("D, H, Dh", "hidden size (32), attention heads (4), per-head size (8)"),
    ("E_t", "dynamic edge tensor (N×N×4)"),
    ("α_ij", "attention weight (also α for miscoverage, by context)"),
    ("α, γ", "target miscoverage (0.10), ACI step size (0.035)"),
    ("ŷ, q_low, q_high", "point forecast and quantile bounds"),
    ("φ, w", "online learner features and weights"),
]:
    c = tbl.add_row().cells
    c[0].text, c[1].text = a, b

out = Path(__file__).resolve().parent / "Mathematical_Model.docx"
doc.save(str(out))
print("saved", out)
