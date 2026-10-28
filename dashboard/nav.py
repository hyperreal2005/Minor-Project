"""Pages, defined once so any view can link to another."""
import streamlit as st

from views import (agreement, datapage, forgetsets, home, inspector, lab, methods, relearning,
                   tour, validity)

HOME = st.Page(home.render, title="The study at a glance", icon=":material/home:",
               url_path="home", default=True)
TOUR = st.Page(tour.render, title="Guided tour", icon=":material/route:", url_path="tour")
INSPECTOR = st.Page(inspector.render, title="Model inspector", icon=":material/search:",
                    url_path="inspector")
VALIDITY = st.Page(validity.render, title="Audit validity", icon=":material/verified:",
                   url_path="validity")
AGREEMENT = st.Page(agreement.render, title="Agreement explorer", icon=":material/scatter_plot:",
                    url_path="agreement")
METHODS = st.Page(methods.render, title="Methods & difficulty", icon=":material/leaderboard:",
                  url_path="methods")
RELEARNING = st.Page(relearning.render, title="Relearning", icon=":material/replay:",
                     url_path="relearning")
FORGETSETS = st.Page(forgetsets.render, title="Forget sets", icon=":material/photo_library:",
                     url_path="forget-sets")
LAB = st.Page(lab.render, title="Calibration lab", icon=":material/tune:", url_path="lab")
DATA = st.Page(datapage.render, title="Data & provenance", icon=":material/table:",
               url_path="data")


def navigation():
    return st.navigation({
        "Overview": [HOME, TOUR],
        "Explore": [INSPECTOR, VALIDITY, AGREEMENT, METHODS, RELEARNING, FORGETSETS],
        "What-ifs": [LAB],
        "Data": [DATA],
    })
