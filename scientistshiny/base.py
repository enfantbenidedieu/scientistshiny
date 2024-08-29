# -*- coding: utf-8 -*-
from shiny import App, run_app
import nest_asyncio
import uvicorn

class Base:
    def __init__(self,model=None):
        pass
    def run(self,**kwargs):
        """
        Run the app
        -----------

        Parameters
        ----------
        kwargs : objet = {}. See https://shiny.posit.co/py/api/App.html
        """
        app = App(ui=self.app_ui, server=self.app_server)
        return run_app(app=app,launch_browser=True,**kwargs)
    
    # Run with notebooks
    def run_notebooks(self,**kwargs):
        """
        Run the app on jupiter notebooks
        --------------------------------
        """
        nest_asyncio.apply()
        uvicorn.run(self.run(**kwargs))
    
    # Stop App
    def stop(self):
        """
        Stop the app
        ------------
        """
        app = App(ui=self.app_ui, server=self.app_server)
        return app.stop()