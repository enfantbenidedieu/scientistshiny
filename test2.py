from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from shiny import App, ui

# first starlette app, just serves static files ----
app_static = StaticFiles(directory=".")

# shiny app ----
app_shiny = App(ui.page_fluid("hello from shiny!"), None)


# combine apps ----
routes = [
    Mount('/static', app=app_static),
    Mount('/shiny', app=app_shiny)
]

app = Starlette(routes=routes)
