from streamlit.testing.v1 import AppTest


BILL_LENGTH_MM_MIN = 20.0
BILL_LENGTH_MM_MAX = 70.0
BILL_DEPTH_MM_MIN = 10.0
BILL_DEPTH_MM_MAX = 30.0
FLIPPER_LENGTH_MM_MIN = 160.0
FLIPPER_LENGTH_MM_MAX = 250.0
BODY_MASS_G_MIN = 2000.0
BODY_MASS_G_MAX = 8000.0
YEAR_MIN = 2007
YEAR_MAX = 2020

NUMINP_BILL_LENGTH_MM_KEY = "bill_length_mm"
NUMINP_BILL_DEPTH_MM_KEY = "bill_depth_mm"
NUMINP_FLIPPER_LENGTH_MM_KEY = "flipper_length_mm"
NUMINP_BODY_MASS_G_KEY = "body_mass_g"
NUMINP_YEAR_KEY = "year"


def test_zero():
    assert True


def test_validation_errors():
    at = AppTest.from_file('app.py').run()
    at.number_input(key=NUMINP_BILL_LENGTH_MM_KEY).set_value(BILL_LENGTH_MM_MIN - 1).run()
    at.button[0].click().run()
    assert at.session_state.is_error_form

    at = AppTest.from_file('app.py').run()
    at.number_input(key=NUMINP_BILL_LENGTH_MM_KEY).set_value(BILL_LENGTH_MM_MAX + 1).run()
    at.button[0].click().run()
    assert at.session_state.is_error_form

    at = AppTest.from_file('app.py').run()
    at.number_input(key=NUMINP_BILL_DEPTH_MM_KEY).set_value(BILL_DEPTH_MM_MIN - 1).run()
    at.button[0].click().run()
    assert at.session_state.is_error_form

    at = AppTest.from_file('app.py').run()
    at.number_input(key=NUMINP_BILL_DEPTH_MM_KEY).set_value(BILL_DEPTH_MM_MAX + 1).run()
    at.button[0].click().run()
    assert at.session_state.is_error_form

    at = AppTest.from_file('app.py').run()
    at.number_input(key=NUMINP_FLIPPER_LENGTH_MM_KEY).set_value(BILL_DEPTH_MM_MIN - 1).run()
    at.button[0].click().run()
    assert at.session_state.is_error_form

    at = AppTest.from_file('app.py').run()
    at.number_input(key=NUMINP_FLIPPER_LENGTH_MM_KEY).set_value(BILL_DEPTH_MM_MAX + 1).run()
    at.button[0].click().run()
    assert at.session_state.is_error_form

    at = AppTest.from_file('app.py').run()
    at.number_input(key=NUMINP_BODY_MASS_G_KEY).set_value(BODY_MASS_G_MIN - 1).run()
    at.button[0].click().run()
    assert at.session_state.is_error_form

    at = AppTest.from_file('app.py').run()
    at.number_input(key=NUMINP_BODY_MASS_G_KEY).set_value(BODY_MASS_G_MAX + 1).run()
    at.button[0].click().run()
    assert at.session_state.is_error_form


def test_complete_result():
    at = AppTest.from_file('app.py').run()
    at.button[0].click().run()
    assert len(at.markdown) == 4
    assert at.markdown[3].value == 'Предполагаемый вид пингвина: Adelie'

