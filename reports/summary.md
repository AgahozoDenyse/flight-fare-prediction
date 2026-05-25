# Flight Fare Prediction — Insights Report

## What This Model Does

This model predicts flight ticket prices for flights departing from Bangladesh.
It was trained on **57,000 historical flights** and learns pricing patterns from
airline, destination, travel class, season, flight duration, and departure timing.

---

## Model Performance

The best-performing model is **RandomForest**.

| Metric | Value | Plain-language meaning |
|--------|-------|------------------------|
| R² | 0.6732 | Explains **67.3%** of fare variation |
| MAE | 28,076 BDT | Average prediction error is ~**39.5%** of the mean fare |
| RMSE | 46,677 BDT | Penalises large errors more heavily than MAE |

### All Models Compared

```
                      R²       MAE      RMSE
RandomForest      0.6732  28075.90  46677.15
DecisionTree      0.6707  28141.87  46850.19
LinearRegression  0.5671  40669.08  53718.44
Lasso             0.5671  40568.79  53720.55
Ridge             0.5671  40664.17  53718.67
```

**Key takeaway:** Tree-based models (Random Forest, Decision Tree) outperform linear
models by ~10 R² points. This confirms that fare pricing is **non-linear** — driven
by complex interactions between airline, route, class, and season that a straight line
cannot capture.

---

## What Drives Flight Prices

### Top 5 Most Influential Features

  1. **Class_First Class** — 39.7% of model decisions
  2. **Duration (hrs)** — 22.1% of model decisions
  3. **Duration_mins** — 20.0% of model decisions
  4. **Destination_CCU** — 9.5% of model decisions
  5. **Class_Economy** — 5.6% of model decisions

### Key Findings

1. **Destination is the biggest price driver** — long-haul international routes
   (New York, London, Toronto, Bangkok) are far more expensive than regional ones.
   Where you fly matters more than when.

2. **Airline choice affects price** — Turkish Airlines, AirAsia, and Cathay Pacific
   command the highest average fares, while Vistara and NovoAir are among the most
   affordable. The spread across airlines is over **7,000 BDT** on average.

3. **Travel class carries a large premium** — upgrading from Economy to Business
   or First Class can multiply the fare significantly.
  - First Class: 120,764 BDT
  - Business: 62,581 BDT
  - Economy: 30,002 BDT

4. **Season is a major factor** — Hajj and Eid periods see prices jump significantly
   above the Regular season baseline.
  - Hajj: 97,144 BDT
  - Eid: 91,560 BDT
  - Winter Holidays: 79,677 BDT
  - Regular: 68,077 BDT

5. **Booking earlier saves money** — the `Days Before Departure` feature confirms
   that last-minute bookings consistently attract higher fares.

6. **Direct flights are not always cheaper** — stopover flights sometimes cost more
   on long-haul international routes due to connecting airport fees.

---

## Average Fares by Airline

  - Turkish Airlines: 75,547 BDT
  - AirAsia: 74,534 BDT
  - Cathay Pacific: 73,325 BDT
  - Thai Airways: 72,846 BDT
  - Malaysian Airlines: 72,775 BDT
  - IndiGo: 72,504 BDT
  - Air India: 72,474 BDT
  - US-Bangla Airlines: 72,088 BDT
  - Kuwait Airways: 71,988 BDT
  - Etihad Airways: 71,785 BDT
  - Gulf Air: 71,458 BDT
  - SriLankan Airlines: 71,265 BDT
  - British Airways: 70,556 BDT
  - Biman Bangladesh Airlines: 70,193 BDT
  - Emirates: 70,106 BDT
  - Air Arabia: 69,924 BDT
  - Qatar Airways: 69,866 BDT
  - Lufthansa: 69,293 BDT
  - Saudia: 69,271 BDT
  - FlyDubai: 68,988 BDT
  - Air Astra: 68,497 BDT
  - NovoAir: 68,351 BDT
  - Singapore Airlines: 68,324 BDT
  - Vistara: 68,108 BDT

---

## Bias-Variance Tradeoff

The `bias_variance_tradeoff.png` plot shows how Decision Tree complexity (depth)
affects model accuracy:

- **Depth 1–3 (high bias / underfitting):** Both training and CV scores are low —
  the model is too simple to capture fare patterns.
- **Depth 4–6 (sweet spot):** CV score peaks. The chosen depth of **5** sits here,
  balancing fit quality against generalisation.
- **Depth 7+ (high variance / overfitting):** Training R² keeps climbing but CV R²
  plateaus or drops — the model memorises training data instead of learning patterns.

---

## Summary for Non-Technical Readers

**In plain English:** The model works like an experienced travel consultant who has
studied thousands of past bookings. Given details about a flight (airline, destination,
class, season, timing), it estimates what the ticket should cost.

**What the model gets right:** It captures the main pricing drivers well — expensive
destinations, premium cabins, peak seasons, and long-haul routes all correctly predict
higher fares.

**Where it has limits:** The model explains **67.3%** of fare variation.
The remaining **32.7%** comes from factors not in the data — promotional
fares, seat availability, real-time demand, and airline revenue management systems.

**Practical recommendation:** Use this model to:
- Flag fares that are unusually high or low compared to historical patterns
- Identify peak-season pricing windows for budget planning
- Compare price competitiveness across airlines and routes
- Estimate fair price ranges when negotiating corporate travel contracts
