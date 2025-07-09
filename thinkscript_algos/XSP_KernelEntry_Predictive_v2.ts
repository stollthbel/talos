# XSP_KernelEntry_Predictive_v2
# Enhanced Kernel Entry System with Predictive Signal Generation
# Designed to catch moves BEFORE they happen using advanced momentum analysis

declare lower;

input lookback = 20;
input fastEMA = 8;
input slowEMA = 21;
input rsiLength = 14;
input kernelLength = 5;
input volumeThreshold = 1.5;
input momentumThreshold = 0.002;

# === CORE INDICATORS ===
def vwapValue = vwap(period = AggregationPeriod.DAY);
def fastEMAValue = ExpAverage(close, fastEMA);
def slowEMAValue = ExpAverage(close, slowEMA);
def rsiValue = RSI(length = rsiLength);
def atr = Average(TrueRange(high, close, low), 14);

# === ADVANCED KERNEL SYSTEM ===
# Multi-timeframe kernel using weighted moving averages
def kernelPrice1 = ExpAverage(close, kernelLength);
def kernelPrice2 = ExpAverage(close, kernelLength * 2);
def kernelPrice3 = ExpAverage(close, kernelLength * 3);

# Composite kernel with momentum weighting
def kernelComposite = (kernelPrice1 * 0.5) + (kernelPrice2 * 0.3) + (kernelPrice3 * 0.2);
def kernelSlope = (kernelComposite - kernelComposite[1]) / kernelComposite[1];
def kernelAcceleration = kernelSlope - kernelSlope[1];

# === VOLUME ANALYSIS ===
def avgVolume = Average(volume, 20);
def volumeRatio = volume / avgVolume;
def volumeThrust = volumeRatio > volumeThreshold;

# === SUPPORT/RESISTANCE DYNAMICS ===
def resistance = Highest(high, lookback)[1];
def support = Lowest(low, lookback)[1];
def midPoint = (resistance + support) / 2;

# Dynamic resistance/support based on volatility
def dynamicResistance = resistance + (atr * 0.5);
def dynamicSupport = support - (atr * 0.5);

# === BREAKOUT PREDICTION LOGIC ===
# Pre-breakout accumulation pattern
def nearResistance = high >= resistance * 0.98 and high <= resistance * 1.02;
def nearSupport = low <= support * 1.02 and low >= support * 0.98;

# Momentum buildup detection
def momentumBuildup = kernelSlope > momentumThreshold and kernelAcceleration > 0;
def momentumFading = kernelSlope < -momentumThreshold and kernelAcceleration < 0;

# Price compression (volatility squeeze)
def priceRange = high - low;
def avgRange = Average(priceRange, 10);
def compression = priceRange < avgRange * 0.7;

# === VWAP REGIME ANALYSIS ===
def vwapSlope = (vwapValue - vwapValue[1]) / vwapValue[1];
def aboveVWAP = close > vwapValue;
def belowVWAP = close < vwapValue;
def vwapMomentum = vwapSlope > 0.0005;
def vwapWeakness = vwapSlope < -0.0005;

# === EMA CLOUD ANALYSIS ===
def emaCloudBull = fastEMAValue > slowEMAValue;
def emaCloudBear = fastEMAValue < slowEMAValue;
def emaCloudExpanding = AbsValue(fastEMAValue - slowEMAValue) > AbsValue(fastEMAValue[1] - slowEMAValue[1]);

# === RSI DIVERGENCE DETECTION ===
def rsiSlope = rsiValue - rsiValue[1];
def priceSlope = close - close[1];
def bullishDivergence = priceSlope < 0 and rsiSlope > 0 and rsiValue < 30;
def bearishDivergence = priceSlope > 0 and rsiSlope < 0 and rsiValue > 70;

# === PREDICTIVE LONG SIGNAL ===
def longSetupBase = emaCloudBull and aboveVWAP and vwapMomentum;
def longMomentumConfirm = momentumBuildup and kernelComposite > kernelComposite[2];
def longVolumeConfirm = volumeThrust;
def longTechnicalConfirm = rsiValue > 45 and rsiValue < 70;

# Pre-breakout long conditions
def longPreBreakout = nearResistance and compression and longMomentumConfirm;
def longDivergencePlay = bullishDivergence and nearSupport;
def longVWAPBounce = close crosses above vwapValue and longMomentumConfirm;

# Master long signal
def explosiveLongEntry = (longSetupBase and longMomentumConfirm and longVolumeConfirm and longTechnicalConfirm) or
                         longPreBreakout or 
                         longDivergencePlay or
                         longVWAPBounce;

# === PREDICTIVE SHORT SIGNAL ===
def shortSetupBase = emaCloudBear and belowVWAP and vwapWeakness;
def shortMomentumConfirm = momentumFading and kernelComposite < kernelComposite[2];
def shortVolumeConfirm = volumeThrust;
def shortTechnicalConfirm = rsiValue < 55 and rsiValue > 30;

# Pre-breakdown short conditions
def shortPreBreakdown = nearSupport and compression and shortMomentumConfirm;
def shortDivergencePlay = bearishDivergence and nearResistance;
def shortVWAPReject = close crosses below vwapValue and shortMomentumConfirm;

# Master short signal
def explosiveShortEntry = (shortSetupBase and shortMomentumConfirm and shortVolumeConfirm and shortTechnicalConfirm) or
                          shortPreBreakdown or 
                          shortDivergencePlay or
                          shortVWAPReject;

# === SIGNAL QUALITY SCORING ===
def longQuality = (if longSetupBase then 0.25 else 0) +
                  (if longMomentumConfirm then 0.25 else 0) +
                  (if longVolumeConfirm then 0.25 else 0) +
                  (if longTechnicalConfirm then 0.25 else 0);

def shortQuality = (if shortSetupBase then 0.25 else 0) +
                   (if shortMomentumConfirm then 0.25 else 0) +
                   (if shortVolumeConfirm then 0.25 else 0) +
                   (if shortTechnicalConfirm then 0.25 else 0);

# === ENHANCED ENTRY TRIGGERS ===
# Only trigger on high-quality setups with momentum confirmation
def qualifiedLongEntry = explosiveLongEntry and longQuality >= 0.75;
def qualifiedShortEntry = explosiveShortEntry and shortQuality >= 0.75;

# === ADAPTIVE PRICE TARGETS ===
def longTarget = close + (atr * 2);
def shortTarget = close - (atr * 2);
def stopLoss = atr * 1.5;

# === PLOTS ===
plot LongEntrySignal = if qualifiedLongEntry then longTarget else Double.NaN;
LongEntrySignal.SetDefaultColor(Color.GREEN);
LongEntrySignal.SetPaintingStrategy(PaintingStrategy.TRIANGLES);
LongEntrySignal.SetLineWeight(3);

plot ShortEntrySignal = if qualifiedShortEntry then shortTarget else Double.NaN;
ShortEntrySignal.SetDefaultColor(Color.RED);
ShortEntrySignal.SetPaintingStrategy(PaintingStrategy.TRIANGLES);
ShortEntrySignal.SetLineWeight(3);

# Support/Resistance levels
plot ResistanceLevel = resistance;
ResistanceLevel.SetDefaultColor(Color.DARK_RED);
ResistanceLevel.SetStyle(Curve.LONG_DASH);

plot SupportLevel = support;
SupportLevel.SetDefaultColor(Color.DARK_GREEN);
SupportLevel.SetStyle(Curve.LONG_DASH);

# Kernel trend line
plot KernelTrend = kernelComposite;
KernelTrend.SetDefaultColor(Color.YELLOW);
KernelTrend.SetLineWeight(2);

# VWAP
plot VWAPLine = vwapValue;
VWAPLine.SetDefaultColor(Color.CYAN);
VWAPLine.SetLineWeight(2);

# Quality indicator
plot QualityLong = if qualifiedLongEntry then longQuality * 100 else Double.NaN;
QualityLong.SetDefaultColor(Color.LIGHT_GREEN);

plot QualityShort = if qualifiedShortEntry then shortQuality * 100 else Double.NaN;
QualityShort.SetDefaultColor(Color.LIGHT_RED);

# === ALERTS ===
Alert(qualifiedLongEntry, "XSP KERNEL LONG ENTRY", Alert.BAR, Sound.Chimes);
Alert(qualifiedShortEntry, "XSP KERNEL SHORT ENTRY", Alert.BAR, Sound.Bell);

# === LABELS ===
AddLabel(qualifiedLongEntry, "LONG KERNEL ENTRY", Color.GREEN);
AddLabel(qualifiedShortEntry, "SHORT KERNEL ENTRY", Color.RED);
AddLabel(compression, "COMPRESSION", Color.YELLOW);
AddLabel(momentumBuildup, "MOMENTUM+", Color.LIGHT_GREEN);
AddLabel(momentumFading, "MOMENTUM-", Color.LIGHT_RED);
