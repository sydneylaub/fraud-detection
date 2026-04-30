from risk_rules import label_risk, score_transaction

# A neutral transaction that contributes 0 to the score on its own.
BASE_TX = {
    "device_risk_score": 0,
    "is_international": 0,
    "amount_usd": 0,
    "velocity_24h": 0,
    "failed_logins_24h": 0,
    "prior_chargebacks": 0,
}


def tx(**overrides):
    return {**BASE_TX, **overrides}


# ---------------------------------------------------------------------------
# label_risk — boundary conditions
# ---------------------------------------------------------------------------

def test_label_risk_low_boundary():
    assert label_risk(0) == "low"
    assert label_risk(29) == "low"


def test_label_risk_medium_boundary():
    assert label_risk(30) == "medium"
    assert label_risk(59) == "medium"


def test_label_risk_high_boundary():
    assert label_risk(60) == "high"
    assert label_risk(100) == "high"


# ---------------------------------------------------------------------------
# score_transaction — device risk signal
# ---------------------------------------------------------------------------

def test_high_device_risk_adds_points():
    """device_risk_score >= 70 must increase the score (was inverted before fix)."""
    assert score_transaction(tx(device_risk_score=70)) == 25
    assert score_transaction(tx(device_risk_score=95)) == 25


def test_medium_device_risk_adds_points():
    assert score_transaction(tx(device_risk_score=40)) == 10
    assert score_transaction(tx(device_risk_score=69)) == 10


def test_low_device_risk_adds_nothing():
    assert score_transaction(tx(device_risk_score=39)) == 0


# ---------------------------------------------------------------------------
# score_transaction — international signal
# ---------------------------------------------------------------------------

def test_international_adds_points():
    """is_international == 1 must increase the score (was inverted before fix)."""
    assert score_transaction(tx(is_international=1)) == 15


def test_domestic_adds_nothing():
    assert score_transaction(tx(is_international=0)) == 0


# ---------------------------------------------------------------------------
# score_transaction — amount signal
# ---------------------------------------------------------------------------

def test_large_amount_adds_points():
    assert score_transaction(tx(amount_usd=1000)) == 25
    assert score_transaction(tx(amount_usd=5000)) == 25


def test_medium_amount_adds_points():
    assert score_transaction(tx(amount_usd=500)) == 10
    assert score_transaction(tx(amount_usd=999)) == 10


def test_small_amount_adds_nothing():
    assert score_transaction(tx(amount_usd=499)) == 0


# ---------------------------------------------------------------------------
# score_transaction — velocity signal
# ---------------------------------------------------------------------------

def test_high_velocity_adds_points():
    """velocity_24h >= 6 must increase the score (was inverted before fix)."""
    assert score_transaction(tx(velocity_24h=6)) == 20
    assert score_transaction(tx(velocity_24h=20)) == 20


def test_medium_velocity_adds_points():
    assert score_transaction(tx(velocity_24h=3)) == 5
    assert score_transaction(tx(velocity_24h=5)) == 5


def test_low_velocity_adds_nothing():
    assert score_transaction(tx(velocity_24h=2)) == 0


# ---------------------------------------------------------------------------
# score_transaction — failed logins signal
# ---------------------------------------------------------------------------

def test_many_failed_logins_adds_points():
    assert score_transaction(tx(failed_logins_24h=5)) == 20
    assert score_transaction(tx(failed_logins_24h=10)) == 20


def test_some_failed_logins_adds_points():
    assert score_transaction(tx(failed_logins_24h=2)) == 10
    assert score_transaction(tx(failed_logins_24h=4)) == 10


def test_no_failed_logins_adds_nothing():
    assert score_transaction(tx(failed_logins_24h=1)) == 0


# ---------------------------------------------------------------------------
# score_transaction — prior chargebacks signal
# ---------------------------------------------------------------------------

def test_multiple_prior_chargebacks_add_points():
    """prior_chargebacks >= 2 must increase the score (was inverted before fix)."""
    assert score_transaction(tx(prior_chargebacks=2)) == 20
    assert score_transaction(tx(prior_chargebacks=5)) == 20


def test_one_prior_chargeback_adds_points():
    """prior_chargebacks == 1 must increase the score (was inverted before fix)."""
    assert score_transaction(tx(prior_chargebacks=1)) == 5


def test_no_prior_chargebacks_adds_nothing():
    assert score_transaction(tx(prior_chargebacks=0)) == 0


# ---------------------------------------------------------------------------
# score_transaction — score clamping
# ---------------------------------------------------------------------------

def test_score_floor_is_zero():
    assert score_transaction(BASE_TX) == 0


def test_score_ceiling_is_100():
    # All high-risk signals fire: 25+15+25+20+20+20 = 125, clamped to 100.
    high_risk = tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=1500,
        velocity_24h=8,
        failed_logins_24h=6,
        prior_chargebacks=3,
    )
    assert score_transaction(high_risk) == 100


# ---------------------------------------------------------------------------
# score_transaction — end-to-end risk classification
# ---------------------------------------------------------------------------

def test_fully_clean_transaction_is_low_risk():
    assert label_risk(score_transaction(BASE_TX)) == "low"


def test_fully_suspicious_transaction_is_high_risk():
    high_risk = tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=1500,
        velocity_24h=8,
        failed_logins_24h=6,
        prior_chargebacks=3,
    )
    assert label_risk(score_transaction(high_risk)) == "high"
