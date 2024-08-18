# Data Rockie - Streamlit 101 - Note
*Patcharanat P.*

- [Original Slide](https://docs.google.com/presentation/d/1__INh0aD_xLmuvKD9pTKXzrvZ_noYodMfaohZuPM7eY/edit#slide=id.g1500a8fe835_0_1)
- Official Streamlit Documentation
    - [Getting Started - Main Concepts](https://docs.streamlit.io/get-started/fundamentals/main-concepts)
    - [API Reference](https://docs.streamlit.io/develop/api-reference)


# Why Streamlit?
- Interactivity
    - interact with widgets and instantly see its effect in a self-serve manner.
- Instant access to the app's functionality.
- `Streamlit` is an open source web framework in Python that turns data scripts into shareable web apps in the minutes
    - Typically, creating web apps require a steep learning curve. But, Streamlit requires no front-end experience, so you can focus on the data and model.
    - Streamlit makes the app creation process as simple as writing Python scripts
- 3 simple principles of streamlit
    1. Embrace scripting
        - Build an app in a few lines of code with simple API. Then see it automatically update as you iteratively save the source file.
    2. Weave in interaction
        - Adding widgets, no need to write a backend, define routes, handle HTTP requests, connect a frontend.
    3. Deploy instantly

# Installation
```bash
pip install streamlit
streamlit hello
```

## Gallery of Streamlit apps
- If you're looking to get inspired, you can check out the `Gallery` for examples of various Streamlit apps across several verticals: https://streamlit.io/gallery

# In Action
```python
# -- app.py
import streamlit as st

st.write("Hello world!")
```

```bash
streamlit run app.py
# This equal to
# python -m streamlit run app.py
```

# Deploy Everywhere
- Locally
- Cloud-host
    - Streamlit Community Cloud
    - HF Spaces
- Self-host
    - GCP, Azure, AWS, etc.
- In-browser
    - Stlite

# Technical Note from Official Documentation
- [Basic Concepts](https://docs.streamlit.io/get-started/fundamentals/main-concepts)
    - Whenever a callback is passed to a widget via the `on_change` (or `on_click`) parameter, the callback will always run before the rest of your script. For details on the Callbacks API, please refer to our [Session State API Reference Guide](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state#use-callbacks-to-update-session-state).

        And to make all of this fast and seamless, Streamlit does some heavy lifting for you behind the scenes. A big player in this story is the [`@st.cache_data`](https://docs.streamlit.io/get-started/fundamentals/main-concepts#caching) decorator, which allows developers to skip certain costly computations when their apps rerun. We'll cover caching later in this page.

    - 