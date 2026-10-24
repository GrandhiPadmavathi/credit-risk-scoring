-- Behavioural features from six monthly statements (m = 1 is the most recent month, m = 6 the oldest).
-- status: -2 no balance, -1 paid in full, 0 revolving credit, 1..9 months payment is overdue.
-- The payment made in month m settles the bill issued in month m + 1.
WITH monthly AS (
    SELECT id, 1 AS m, pay_1 AS status, bill_amt1 AS bill, pay_amt1 AS paid FROM applicants
    UNION ALL SELECT id, 2, pay_2, bill_amt2, pay_amt2 FROM applicants
    UNION ALL SELECT id, 3, pay_3, bill_amt3, pay_amt3 FROM applicants
    UNION ALL SELECT id, 4, pay_4, bill_amt4, pay_amt4 FROM applicants
    UNION ALL SELECT id, 5, pay_5, bill_amt5, pay_amt5 FROM applicants
    UNION ALL SELECT id, 6, pay_6, bill_amt6, pay_amt6 FROM applicants
),
statements AS (
    SELECT *, LEAD(bill) OVER (PARTITION BY id ORDER BY m) AS bill_settled
    FROM monthly
),
behaviour AS (
    SELECT
        id,
        MAX(GREATEST(status, 0))                                    AS max_delay,
        MAX(CASE WHEN m = 1 THEN GREATEST(status, 0) END)           AS recent_delay,
        COUNT(*) FILTER (WHERE status >= 1)                         AS months_delayed,
        COUNT(*) FILTER (WHERE status >= 1 AND m <= 3)              AS months_delayed_last3,
        COUNT(*) FILTER (WHERE status = -1)                         AS months_paid_full,
        COUNT(*) FILTER (WHERE status = -2)                         AS months_no_balance,
        MAX(CASE WHEN m = 1 THEN bill END)                          AS last_bill,
        AVG(bill)                                                   AS avg_bill,
        MAX(bill)                                                   AS max_bill,
        AVG(paid)                                                   AS avg_payment,
        MAX(CASE WHEN m = 1 THEN paid END)                          AS last_payment,
        COUNT(*) FILTER (WHERE paid = 0 AND bill_settled > 0)       AS months_no_payment,
        SUM(paid) FILTER (WHERE bill_settled IS NOT NULL)
            / NULLIF(SUM(GREATEST(bill_settled, 0)), 0)             AS pay_to_bill_ratio,
        REGR_SLOPE(bill, 7 - m)                                     AS bill_slope
    FROM statements
    GROUP BY id
)
SELECT
    a.id,
    a.limit_bal,
    b.max_delay,
    b.recent_delay,
    b.months_delayed,
    b.months_delayed_last3,
    b.months_paid_full,
    b.months_no_balance,
    b.months_no_payment,
    b.last_bill / a.limit_bal                                       AS utilization,
    b.avg_bill / a.limit_bal                                        AS avg_utilization,
    b.max_bill / a.limit_bal                                        AS max_utilization,
    LEAST(COALESCE(b.pay_to_bill_ratio, 1.0), 5.0)                  AS pay_to_bill_ratio,
    b.last_payment / a.limit_bal                                    AS last_payment_to_limit,
    b.avg_payment / a.limit_bal                                     AS avg_payment_to_limit,
    COALESCE(b.bill_slope, 0) / a.limit_bal                         AS bill_trend
FROM applicants a
JOIN behaviour b USING (id)
ORDER BY a.id
