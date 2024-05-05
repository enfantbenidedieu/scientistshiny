from shiny import App, ui

from itables.sample_dfs import get_countries
from itables import to_html_datatable

df = get_countries()

app_ui = ui.page_fluid(ui.HTML(to_html_datatable(df)))

app = App(app_ui, server=None)
app.run()