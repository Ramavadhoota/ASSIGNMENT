Bitcoin Market Sentiment vs Trader Performance Analysis
Primetrade.ai — Data Science Internship · Round 0

Analysing how Bitcoin Fear/Greed Index sentiment (2018–2025) relates to trader behavior and performance on Hyperliquid, using 211,224 trade fills from 32 unique accounts.


Table of Contents

Project Overview
Datasets
Setup & Usage
Methodology
Key Findings
Trader Segmentation
Predictive Model
Strategy Recommendations
Output Files


Project Overview
Question: Does Bitcoin market sentiment (Fear vs Greed) meaningfully predict or correlate with how traders perform on Hyperliquid?
Short answer: Not in a straightforward way. The analysis reveals a more nuanced story — sentiment shapes how traders behave (sizing, directionality, activity), but the link to outcomes is weaker than expected and not statistically significant for most metrics. The real performance differentiator is trader segment, not sentiment.
Tech stack: Python · pandas · scikit-learn · scipy · seaborn/matplotlib · Google Colab

Datasets
FileRowsPeriodKey Columnsfear_greed_index.csv2,645Feb 2018 – May 2025date, value (0–100), classificationhistorical_data.csv211,224Dec 2024Account, Closed PnL, Size USD, Side, Direction, Timestamp IST
Sentiment Class Distribution (full history)
ClassDaysScore RangeExtreme Fear~5500–24Fear~78025–44Neutral~43045–54Greed~58055–74Extreme Greed~30575–100
Critical Data Note — Open vs Close Fills
On Hyperliquid, every position has two fills: one when it opens (Closed PnL = 0) and one when it closes (Closed PnL ≠ 0). Analysing all fills naively would corrupt PnL metrics with thousands of zeros. This pipeline separates them:

All fills → used for volume, frequency, and fee metrics
Closed fills only → used for PnL, win rate, and risk metrics


Setup & Usage
1. Open in Google Colab
Upload untitled33.py as a notebook or paste into a new Colab cell.
2. Upload data files
/content/fear_greed_index.csv
/content/historical_data.csv
3. Run all cells top to bottom
Outputs are saved to /content/outputs/ and auto-downloaded as a zip at the end.
Dependencies (auto-installed)
pandas  numpy  matplotlib  seaborn  scipy  scikit-learn

Methodology
Data Pipeline
Raw CSVs
   ↓
Quality checks (missing values, duplicates, type casting)
   ↓
Separate OPEN fills from CLOSE fills (Direction column)
   ↓
Clip Closed PnL outliers at 1st–99th percentile
   ↓
Parse Timestamp IST (DD-MM-YYYY HH:MM, dayfirst=True)
   ↓
Map 5 sentiment classes → binary (Fear / Neutral / Greed)
   ↓
Left-merge on date_only
   ↓
Daily per-account aggregation (two tables: all fills + closed fills)
   ↓
Analysis → Segmentation → Model
Sentiment Mapping
Raw ClassificationBinary LabelBinary for Stats TestsExtreme FearFearFearFearFearFearNeutralNeutral(excluded from Fear vs Greed t-tests)GreedGreedGreedExtreme GreedGreedGreed
Daily Metrics Computed
MetricSourceDescriptionnum_closed_tradesClosed fillsTrade count per account per daytotal_pnlClosed fillsGross realised PnLnet_pnlClosed fills - feesPnL after trading feeswin_rateClosed fillsFraction of profitable closesavg_trade_size_usdClosed fillsAverage notional per tradedrawdown_proxyClosed fillsIntraday PnL standard deviationlong_pct_closedClosed fillsFraction of long-side closestotal_feesAll fillsSum of fees paid

Key Findings
Finding 1 — Sentiment Has No Statistically Significant Effect on Performance
The Mann-Whitney U test (two-sided, α = 0.05) found no significant difference between Fear and Greed days for any tested metric.
MetricFear MeanGreed MeanΔp-valueSignificant?Daily PnL ($)~$5,100~$4,200−$9000.1004❌ NoWin Rate~84%~85%+1%0.2478❌ NoClosed Trades/Day———0.4514❌ NoPnL Std Dev———0.4586❌ NoAvg Trade Size (USD)———0.2223❌ NoNet PnL after Fees———0.0747❌ No

Interpretation: This is a meaningful result in itself — it means traders in this dataset cannot rely on sentiment as a reliable signal for when to trade. Other factors dominate performance variance.


Finding 2 — Trade Size and Activity Decrease Monotonically from Fear → Greed
The behavior heatmap reveals a clear structural pattern across all 5 sentiment classes:
MetricExtreme FearFearNeutralGreedExtreme GreedAvg Closed Trades/Day81.9467.0065.8356.2852.68Avg Trade Size (USD)$13,233$11,126$8,576$7,196$6,401Win Rate0.770.860.830.850.87Total PnL ($)4,6735,3324,0163,7894,620Total Fees ($)178.78194.14137.69120.4760.62
Traders are most active and deploy the largest sizes on Fear days — the opposite of conventional wisdom. During Extreme Greed, activity and sizing drop significantly. This could indicate professionals fading rallies and accumulating on fear, while retail sentiment drives the headline index.

Finding 3 — Trade Size Is Negatively Correlated with FG Score (Statistically Significant)
Continuous correlation analysis with the numeric Fear/Greed score (0–100):
MetricPearson rp-valueInterpretationDaily PnL−0.0060.8006No relationshipWin Rate+0.0440.0683Marginal positive trendAvg Trade Size (USD)−0.0800.0010Significant: larger trades on Fear days
The only statistically significant continuous relationship is trade size declining as sentiment becomes more greedy — reinforcing Finding 2.

Finding 4 — Traders Are Short-Biased Across All Sentiment Classes
Long trade fraction never exceeds 50% in any sentiment class:
SentimentLong %Extreme Fear34.9%Fear40.1%Neutral37.6%Greed40.9% ← peakExtreme Greed36.1%
The community skews short throughout — suggesting systematic short strategies or hedging dominates this account set. The conventional "greed = long" narrative is not reflected here.

Finding 5 — Neutral Days Sit Closer to Fear in Activity, Closer to Greed in Size
Neutral days show mid-range trade counts (65.83) and mid-range trade sizes ($8,576), sitting between the Fear and Greed extremes on most metrics. They should not be treated as a default "normal" — they inherit slightly elevated risk (fees: $137.69) without a corresponding uplift in returns.

Trader Segmentation
32 Unique Accounts — 4-Quadrant Rule-Based Segmentation
Segmented on median splits of: avg daily trades (frequency) and lifetime PnL (profitability).
SegmentTradersAvg Daily TradesLifetime PnLAvg Win RateFee DragFrequent / Profitable10HighHigh~83%LowFrequent / Loss-Making6HighLow~84%Very HighInfrequent / Profitable6LowHigh~87%LowInfrequent / Loss-Making10LowLow~80%Low
Key observation: Win rates are remarkably similar across all 4 segments (80–87%). The performance gap is driven by fee drag, not skill. The Frequent/Loss-Making segment has a wide, variable fee drag (up to 45% of |PnL|), confirming that overtrading erodes a viable edge.
The top traders by lifetime PnL (Infrequent/Profitable, Frequent/Profitable) achieved $400k–$900k+ in lifetime PnL.
K-Means Clustering (k=3)
Three behavioral archetypes identified via unsupervised clustering on scaled features:
ClusternProfileCluster 09Mid-frequency, moderate win rate (~83–91%)Cluster 17Low-to-mid frequency, highest variability in win rateCluster 216Largest group; wide range of trade frequencies (10–440/day)

Predictive Model
Task: Predict whether a trader will have a profitable day tomorrow, given today's behavior and sentiment.
Target: next_day_profitable (binary: tomorrow's PnL > 0)
Features

lag1_total_pnl — previous day's gross PnL
lag1_drawdown_proxy — previous day's risk (PnL std dev)
lag1_avg_trade_size_usd — previous day's avg trade size
lag1_total_fees — previous day's fee spend
lag1_num_closed_trades — previous day's trade count
fg_score_lag1 — previous day's Fear/Greed score (0–100)
lag1_win_rate — previous day's win rate
lag1_long_pct_closed — previous day's long bias
sentiment_today_enc — today's sentiment (0=Fear, 1=Neutral, 2=Greed)

Results
ModelTest AccuracyROC-AUCCV-AUCLogistic Regression———Random ForestBestBest—Gradient Boosting———
Best model: Random Forest
Feature Importance (Random Forest)
The model confirms what the exploratory analysis found — sentiment is a minor predictor:

lag1_total_pnl ← most important (yesterday's PnL predicts tomorrow's)
lag1_drawdown_proxy (risk proxy)
lag1_avg_trade_size_usd (position sizing)
lag1_total_fees (fee activity)
lag1_num_closed_trades (trade count)
fg_score_lag1 (Fear/Greed score)
lag1_win_rate
lag1_long_pct_closed
sentiment_today_enc ← least important


⚠️ Model caveat: The dataset is highly imbalanced — profit days dominate. The confusion matrix shows the model predicts "Profit Day" for nearly all cases (272 correct, 42 false positives, 0 true negatives). This suggests the model has learned the base rate, not a true signal. A larger, more balanced dataset would be needed for a production-grade classifier.


Strategy Recommendations
Strategy 1 — Sentiment-Responsive Risk & Fee Management
Evidence: Trade size is significantly larger on Fear days; fees are higher too. Managing size and fees matters more than timing sentiment.
SentimentPosition SizeDaily Trade CapFee RuleExtreme FearBaseline (already elevated)75% of normalTrack cumulative fees closelyFearBaselineBaselineStandardNeutralBaselineBaselineTighten stopsGreedBaselineBaselineHarvest profits 10–15% earlierExtreme Greed−20%BaselineConsider short hedge if long% > 65%
Strategy 2 — Segment-Tailored Playbooks
SegmentPriorityRoot CauseFear PlaybookGreed PlaybookFrequent / ProfitableMonitorNone — preserve edgeHalve leverage; maintain count+10% allocation; watch fee accumulationFrequent / Loss-Making🔴 HIGH INTERVENTIONFee drag + overtradingMax 3 trades/day; counter-trend onlyMax 1 directional trade; rest in stablesInfrequent / ProfitableMonitorNone — maintain discipline+15% size on top setupsStandard size; early profit harvestInfrequent / Loss-Making🟡 TRAINING NEEDEDNo systematic processPaper-trade / 0.1% risk only3–4 rule-based trades; 0.5% risk each
Strategy 3 — Short-Bias Awareness & Contrarian Overlay
Given that traders are consistently short-biased (max long% = 40.9% at Greed):

Signal: When long% < 35% + FG Score < 25 → community is maximally short + fearful. Potential long accumulation opportunity in 3 tranches.
Signal: When long% > 41% + FG Score > 75 → crowd is relatively long at Greed peak. Reduce short exposure; avoid chasing.


Output Files
FileDescription01_metric_distributions.pngHistograms of 6 key daily metrics with mean/median lines02_sentiment_overview.pngSentiment class bar chart + FG score fill chart (2023–present)03_timeseries_overview.pngDaily PnL / trade count / win rate time-series coloured by sentiment04_fear_vs_greed.pngBox plots comparing Fear vs Greed across 6 metrics with p-values05_behavior_heatmap.pngNormalised heatmap of 7 metrics across all 5 sentiment classes06_fg_score_scatter.pngPearson correlation scatter: FG score (0–100) vs PnL, win rate, trade size07_directional_bias.pngLong % by sentiment class + trade size violin plots08_segmentation.png4-quadrant scatter, win rate bars, fee drag box, segment counts09_kmeans.pngElbow method + K-Means (k=3) cluster scatter10_insight_evidence.pngSummary evidence for all 5 insights11_model_evaluation.pngConfusion matrix + Random Forest feature importancesaction_matrix.csvSegment × Sentiment action playbook (machine-readable)

Reproducibility
ItemDetailRandom seed42 throughoutTrain/test split80/20, stratifiedOutlier treatment1st–99th percentile clip on Closed PnLTimestamp parsingdayfirst=True for DD-MM-YYYY HH:MM IST formatSentiment mergeLeft join on date_only (day-level)Open/close filterDirection.contains('CLOSE') OR Closed PnL != 0

Primetrade.ai · Data Science Internship Assignment · Round 0
