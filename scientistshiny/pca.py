# -*- coding: utf-8 -*-
from shiny import App, Inputs, Outputs, Session, render, ui, reactive, run_app
import shinyswatch

import numpy as np
import pandas as pd
import plotnine as pn
import matplotlib.colors as mcolors
import nest_asyncio
import uvicorn

from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import great_tables as gt
from scientisttools import fviz_pca_ind,fviz_pca_var,fviz_eig,fviz_contrib,fviz_cos2,dimdesc

from .function import *

colors = mcolors.CSS4_COLORS
colors["cos2"] = "cos2"
colors["contrib"] = "contrib"

css_path = Path(__file__).parent / "www" / "style.css"

class PCAshiny(BaseEstimator,TransformerMixin):
    """
    Principal Component Analysis (PCA) with scientistshiny
    ------------------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Principal Component Analysis (PCA) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables on a Shiny for Python application. Allows to change PCA graphical parameters. Graphics can be downloaded in png, jpg and pdf.

    Usage
    -----
    ```python
    >>> PCAshiny(model)
    ```

    Parameters
    ----------
    `model`: an object of class PCA. A PCA result from scientisttools.

    Returns
    -------
    `Graphs` : a tab containing the individuals factor map and the variables factor map

    `Values` : a tab containing the eigenvalue, the results for the variables, the results for the individuals, the results for the supplementary variables and the results for the categorical variables.

    `Automatic description of axes` : a tab containing the output of the dimdesc function. This function is designed to point out the variables and the categories that are the most characteristic according to each dimension obtained by a Factor Analysis.

    `Summary of dataset` : A tab containing the summary of the dataset and a boxplot and histogramm for quantitative variables.

    `Data` : a tab containing the dataset with a nice display.

    The left part of the application allows to change some elements of the graphs (axes, variables, colors,.)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> import pandas as pd
    >>> from scientisttools import PCA, load_decathlon2
    >>> from scientistshiny import PCAshiny
    >>> decathlon = load_decathlon2()
    >>> res_pca = PCA(standardize=True,ind_sup=list(range(23,27)),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(decathlon)
    >>> app = PCAshiny(model=res_pca)
    >>> app.run()
    ```
    
    for jupyter notebooks
    https://stackoverflow.com/questions/74070505/how-to-run-fastapi-application-inside-jupyter
    """
    def __init__(self,model=None):
        # Check if model is Principal Components Analysis (PCA)
        if model.model_ != "pca":
            raise TypeError("'model' must be an object of class PCA")
        
        # Individuals test color
        ind_text_color_choices = {"actif/sup":"actifs/supplémentaires","cos2":"Cosinus","contrib":"Contribution","var_quant":"Variable quantitative","kmeans" : "KMeans"}

        # resume choice
        resume_choices = {"stats_desc":"Statistiques descriptives","hist" : "Histogramme","corr_matrix": "Matrice des corrélations"}

        # Quantitatives columns
        var_labels = model.call_["X"].columns.tolist()
            
        # Initialise value choice
        value_choice = {"eigen_res":"Valeurs propres","var_res":"Résultats des variables","ind_res":"Résultats sur les individus"}

        # Check if supplementary individuals
        if hasattr(model, "ind_sup_"):
            value_choice = {**value_choice, **{"ind_sup_res" : "Résultats des individus supplémentaires"}}
        
        # Check if supplementary quantitatives variables
        if hasattr(model, "quanti_sup_"):
            value_choice = {**value_choice, **{"quanti_sup_res" : "Résultats sur les variables quantitatives supplémentaires"}}
            var_labels = [*var_labels,*model.quanti_sup_["coord"].index.tolist()]
        
        # Check if supplementary qualitatives variables
        if hasattr(model, "quali_sup_"):
            value_choice = {**value_choice, **{"quali_sup_res" : "Résultats des variables qualitatives supplémentaires"}}
            ind_text_color_choices = {**ind_text_color_choices, **{"var_qual":"Variable qualitative"}}
            resume_choices = {**resume_choices, **{"bar_plot":"Diagramme en barres"}}
        
        # UI
        app_ui = ui.page_fluid(
            ui.include_css(css_path),
            shinyswatch.theme.superhero(),
            header(title="Analyse en Composantes Principales",model_name="PCA",background_color="#2e4053"), # #34495e
            ui.page_sidebar(
                ui.sidebar(
                    ui.panel_well(
                        ui.h6("Options graphiques",style="text-align:center"),
                        ui.div(ui.h6("Axes"),style="display: inline-block;padding: 5px"),
                        ui.div( ui.input_select(id="axis1",label="",choices={x:x for x in range(model.call_["n_components"])},selected=0,multiple=False),style="display: inline-block;"),
                        ui.div( ui.input_select(id="axis2",label="",choices={x:x for x in range(model.call_["n_components"])},selected=1,multiple=False),style="display: inline-block;"),
                        ui.br(),
                        ui.div(graph_input(id="fviz_ind_var",choices={"fviz_ind":"individus","fviz_var":"variables"}),style="display: inline-block;"),
                        ui.panel_conditional("input.fviz_ind_var === 'fviz_ind'",
                            title_input(id="ind_title",value="Individuals - PCA"),
                            ui.output_ui("choix_ind_mod"),
                            ui.output_ui("point_label"),
                            text_size_input(which="ind"),
                            point_select_input(id="ind_point"),
                            ui.panel_conditional("input.ind_point === 'cos2'",ui.div(lim_cos2(id="ind_lim_cos2"),align="center")),
                            ui.panel_conditional("input.ind_point === 'contrib'",ui.div(lim_contrib(id="ind_lim_contrib"),align="center")              ),
                            text_color_input(id="ind_text_color",choices=ind_text_color_choices),
                            ui.panel_conditional("input.ind_text_color ==='actif/sup'",
                                ui.input_select(id="ind_text_actif_color",label="Mndividus actifs",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                                ui.output_ui("ind_text_sup_color_choice"),
                                ui.output_ui("ind_text_quali_sup_color_choice")
                            ),
                            ui.panel_conditional("input.ind_text_color === 'kmeans'",ui.input_numeric(id="ind_text_kmeans_nb_clusters",label="Choix du nombre de clusters",value=2,min=1,max=model.ind_["coord"].shape[0],step=1,width="100%")),
                            ui.panel_conditional("input.ind_text_color === 'var_quant'",ui.input_select(id="ind_text_var_quant",label="choix de la variable",choices={x:x for x in var_labels},selected=var_labels[0],width="100%")),
                            ui.panel_conditional("input.ind_text_color === 'var_qual'",ui.output_ui("ind_text_var_qual")),
                            ui.input_switch(id="ind_plot_repel",label="repel",value=True)
                        ),
                        ui.panel_conditional("input.fviz_ind_var === 'fviz_var'",
                            title_input(id="var_title",value="Correlation circle - PCA"),
                            text_size_input(which="var"),
                            point_select_input(id="var_point"),
                            ui.panel_conditional("input.var_point === 'cos2'",ui.div(lim_cos2(id="var_lim_cos2"),align="center")              ),
                            ui.panel_conditional("input.Var_point === 'contrib'",ui.div(lim_contrib(id="var_lim_contrib"),align="center")              ),
                            ui.input_select(id="var_text_color",label="Colorier les flèches par :",choices={"actif/sup":"actives/supplémentaires","cos2":"Cosinus","contrib":"Contribution","kmeans" : "KMeans"},selected="actif/sup",multiple=False,width="100%"),
                            ui.panel_conditional("input.var_text_color === 'actif/sup'",
                                ui.input_select(id="var_text_actif_color",label="Variables actives",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                                ui.output_ui("var_text_sup_color_choice")
                            ),
                            ui.panel_conditional("input.var_text_color === 'kmeans'",ui.input_numeric(id="var_text_kmeans_nb_clusters",label="Choix du nombre de clusters",value=2,min=1,max=model.var_["coord"].shape[0],step=1,width="100%")),
                        )
                    ),
                    ui.div(ui.input_action_button(id="exit",label="Quitter l'application",style='padding:5px; background-color: #fcac44;text-align:center;white-space: normal;'),align="center"),
                    width="25%"
                ),
                ui.navset_card_tab(
                    ui.nav_panel("Graphes",
                        ui.row(
                            ui.column(7,
                                ui.div(ui.output_plot("fviz_ind_plot",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px"),
                                ui.div(ui.download_button(id="download_ind_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_ind_plot_png",label="png",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_ind_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                align="center"
                            ),
                            ui.column(5,
                                ui.div(ui.output_plot("fviz_var_plot",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                                ui.div(ui.download_button(id="download_var_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_var_plot_png",label="png",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_var_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                align="center"
                            )
                        )
                    ),
                    ui.nav_panel("Valeurs",
                        ui.input_radio_buttons(id="value_choice",label=ui.h6("Quelles sorties voulez-vous?"),choices=value_choice,inline=True),
                        ui.br(),
                        eigen_panel(),
                        ui.panel_conditional("input.value_choice === 'var_res'",
                            ui.input_radio_buttons(id="var_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","contrib":"Contributions","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                            ui.panel_conditional("input.var_choice === 'coord'",PanelConditional1(text="var",name="coord")),
                            ui.panel_conditional("input.var_choice === 'contrib'",PanelConditional2(text="var",name="contrib")),
                            ui.panel_conditional("input.var_choice === 'cos2'",PanelConditional2(text="var",name="cos2"))
                        ),
                        ui.panel_conditional("input.value_choice === 'ind_res'",
                            ui.input_radio_buttons(id="ind_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","contrib":"Contributions","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                            ui.panel_conditional("input.ind_choice === 'coord'",PanelConditional1(text="ind",name="coord")),
                            ui.panel_conditional("input.ind_choice === 'contrib'",PanelConditional2(text="ind",name="contrib")),
                            ui.panel_conditional("input.ind_choice === 'cos2'",PanelConditional2(text="ind",name="cos2"))
                        ),
                        ui.output_ui("ind_sup_panel"),
                        ui.output_ui("quanti_sup_panel"),  
                        ui.output_ui("quali_sup_panel")
                    ),
                    dim_desc_panel(model=model),
                    ui.nav_panel("Résumé du jeu de données",
                        ui.input_radio_buttons(id="resume_choice",label=ui.h6("Que voulez - vous afficher?"),choices=resume_choices,selected="stats_desc",width="100%",inline=True),
                        ui.br(),
                        ui.panel_conditional("input.resume_choice === 'stats_desc'",PanelConditional1(text="stats",name="desc")),
                        ui.panel_conditional("input.resume_choice === 'hist'",
                            ui.row(
                                ui.column(2,
                                    ui.input_select(id="var_label",label=ui.h6("Choisir une variable"),choices={x:x for x in var_labels},selected=var_labels[0]),
                                    ui.input_switch(id="add_density",label="Densite",value=False)
                                ),
                                ui.column(10,
                                    ui.div(ui.output_plot("fviz_hist_plot",width='100%', height='500px'),align="center"),
                                    ui.hr(),
                                    ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px"),
                                    ui.div(ui.download_button(id="download_hist_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                    ui.div(ui.download_button(id="download_hist_plot_png",label="png",style = download_btn_style),style="display: inline-block;"),
                                    ui.div(ui.download_button(id="download_hist_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                    align="center"
                                )
                            )
                        ),
                        ui.panel_conditional("input.resume_choice === 'corr_matrix'",PanelConditional1(text="corr",name="matrix")),
                        ui.output_ui("quali_sup_graph")
                    ),
                    ui.nav_panel("Données",PanelConditional1(text="overall",name="data"))
                ),
                class_="bslib-page-dashboard",
                fillable=True
            )
        )

        # Server
        def server(input:Inputs, output:Outputs, session:Session):

            # -------------------------------------------------------
            @reactive.Effect
            @reactive.event(input.exit)
            async def _():
                await session.close()
            
            #----------------------------------------------------------------------------------------------
            # Disable x and y axis
            @reactive.Effect
            def _():
                x = int(input.axis1())
                Dim = [i for i in range(model.call_["n_components"]) if i > x]
                ui.update_select(id="axis2",label="",choices={x : x for x in Dim},selected=Dim[0])
            
            @reactive.Effect
            def _():
                x = int(input.axis2())
                Dim = [i for i in range(model.call_["n_components"]) if i < x]
                ui.update_select(id="axis1",label="",choices={x : x for x in Dim},selected=Dim[0])
            
            #--------------------------------------------------------------------------------------------------
            # Add Supplementary individuals colors choice
            if hasattr(model,"ind_sup_"):
                @render.ui
                def ind_text_sup_color_choice():
                    return ui.TagList(ui.input_select(id="ind_text_sup_color",label="Individus supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="blue",multiple=False,width="100%"))
            
            #---------------------------------------------------------------------------------------------------
            # Add supplementary categories colors choice
            if hasattr(model,"quali_sup_"):
                @render.ui
                def ind_text_quali_sup_color_choice():
                    return ui.TagList(ui.input_select(id="ind_text_quali_sup_color",label="Modalités",choices={x:x for x in mcolors.CSS4_COLORS},selected="red",multiple=False,width="100%"))

            #--------------------------------------------------------------------------------------------------------------
            # Disable individuals colors
            if hasattr(model,"ind_sup_") and hasattr(model,"quali_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_actif_color",label="Individus actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.ind_text_sup_color(),input.ind_text_quali_sup_color()]]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_sup_color",label="Individus supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.ind_text_actif_color(),input.ind_text_quali_sup_color()]]},selected="blue")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_quali_sup_color",label="Modalités supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.ind_text_actif_color(),input.ind_text_sup_color()]]},selected="red")
            elif hasattr(model,"ind_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_actif_color",label="Individus actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.ind_text_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_sup_color",label="Individus supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.ind_text_actif_color()]},selected="blue")
            elif hasattr(model,"quali_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_actif_color",label="Individus actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.ind_text_quali_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_quali_sup_color",label="Modalités supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.ind_text_actif_color()]},selected="red")

            #-------------------------------------------------------------------------------------------------
            if hasattr(model,"quali_sup_"):
                @render.ui
                def ind_text_var_qual():
                    quali_sup_labels = model.quali_sup_["eta2"].index.tolist()
                    return ui.TagList(
                            ui.input_select(id="ind_text_var_qual_color",label="Choix de la variable",choices={x:x for x in quali_sup_labels},selected=quali_sup_labels[0],multiple=False,width="100%"),
                            ui.input_switch(id="ind_text_add_ellipse",label="Trace les ellipses de confiance autour des modalités",value=False)
                        )
            
            #-----------------------------------------------------------------------------------------------
            # Add supplementary quantitatives variables colors choice
            if hasattr(model, "quanti_sup_"):
                @render.ui
                def var_text_sup_color_choice():
                        return ui.TagList(ui.input_select(id="var_text_sup_color",label="Variables supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="blue",multiple=False,width="100%"))
                
                # Disable quantitative variable colors
                @reactive.Effect
                def _():
                    ui.update_select(id="var_text_actif_color",label="Variables quantitatives actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.var_text_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="var_text_sup_color",label="Variables quantitatives supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.var_text_actif_color()]},selected="blue")
                    
            
            #--------------------------------------------------------------------------------
            ## Individuals Factor Map
            #--------------------------------------------------------------------------------
            # Reactive individuals plot
            @reactive.Calc
            def plot_ind():
                # Define boolean
                if hasattr(model,"ind_sup_"):
                    ind_sup = True
                else:
                    ind_sup = False
                    
                if hasattr(model,"quali_sup_"):
                    quali_sup = True
                else:
                    quali_sup = False

                if input.ind_text_color() == "actif/sup":
                    # Define colors for supplementary individuals
                    if hasattr(model,"ind_sup_"):
                        color_sup = input.ind_text_sup_color()
                    else:
                        color_sup = None
                    
                    # Define colors for supplementary modalities
                    if hasattr(model,"quali_sup_"):
                        color_quali_sup = input.ind_text_quali_sup_color()
                    else:
                        color_quali_sup = None
                    
                    fig = fviz_pca_ind(
                            self=model,
                            axis=[int(input.axis1()),int(input.axis2())],
                            color=input.ind_text_actif_color(),
                            ind_sup=ind_sup,
                            color_sup = color_sup,
                            quali_sup=quali_sup,
                            color_quali_sup=color_quali_sup,
                            text_size = input.ind_text_size(),
                            lim_contrib =input.ind_lim_contrib(),
                            lim_cos2 = input.ind_lim_cos2(),
                            title = input.ind_title(),
                            repel=input.ind_plot_repel()
                            )
                elif input.ind_text_color() in ["cos2","contrib"]:
                    fig = fviz_pca_ind(
                            self=model,
                            axis=[int(input.axis1()),int(input.axis2())],
                            color=input.ind_text_color(),
                            ind_sup=ind_sup,
                            quali_sup=quali_sup,
                            text_size = input.ind_text_size(),
                            lim_contrib = input.ind_lim_contrib(),
                            lim_cos2 = input.ind_lim_cos2(),
                            title = input.ind_title(),
                            repel=input.ind_plot_repel()
                        )
                elif input.ind_text_color() == "var_quant":
                    fig = fviz_pca_ind(
                            self=model,
                            axis=[int(input.axis1()),int(input.axis2())],
                            color=input.ind_text_var_quant(),
                            ind_sup=ind_sup,
                            quali_sup=quali_sup,
                            text_size = input.ind_text_size(),
                            lim_contrib =input.ind_lim_contrib(),
                            lim_cos2 = input.ind_lim_cos2(),
                            title = input.ind_title(),
                            legend_title=input.ind_text_var_quant(),
                            repel=input.ind_plot_repel()
                        )
                elif input.ind_text_color() == "kmeans":
                    kmeans = KMeans(n_clusters=input.ind_text_kmeans_nb_clusters(), random_state=np.random.seed(123), n_init="auto").fit(model.ind_["coord"])
                    fig = fviz_pca_ind(
                            self = model,
                            axis = [int(input.axis1()),int(input.axis2())],
                            color = kmeans,
                            ind_sup = ind_sup,
                            quali_sup = quali_sup,
                            text_size = input.ind_text_size(),
                            lim_contrib = input.ind_lim_contrib(),
                            lim_cos2 = input.ind_lim_cos2(),
                            title = input.ind_title(),
                            repel=input.ind_plot_repel()
                        )
                elif input.ind_text_color() == "var_qual":
                    fig = fviz_pca_ind(
                            self=model,
                            axis=[int(input.axis1()),int(input.axis2())],
                            ind_sup=ind_sup,
                            quali_sup=quali_sup,
                            text_size = input.ind_text_size(),
                            lim_contrib =input.ind_lim_contrib(),
                            lim_cos2 = input.ind_lim_cos2(),
                            title = input.ind_title(),
                            habillage= input.ind_text_var_qual_color(),
                            add_ellipses=input.ind_text_add_ellipse(),
                            repel=input.ind_plot_repel()
                        )
                return fig+pn.theme_gray()

            # ------------------------------------------------------------------------------
            # Individual Factor Map - PCA
            @render.plot(alt="Individuals Factor Map - PCA")
            def fviz_ind_plot():
                return plot_ind().draw()
            
            # Downlaod
            # @render.download(filename="Individuals-Factor-Map.png")
            # def download_ind_plot_png():
            #    return plot_ind().save("Individuals-Factor-Map.png")
            
            #--------------------------------------------------------------------------------------------
            ## Variables Factor Map
            #--------------------------------------------------------------------------------------------
            @reactive.Calc
            def plot_var():
                # Define boolean 
                if hasattr(model, "quanti_sup_"):
                    quanti_sup = True
                else:
                    quanti_sup = False
                if input.var_text_color() == "actif/sup":
                    if hasattr(model, "quanti_sup_"):
                        color_sup = input.var_text_sup_color()
                    else:
                        color_sup = None
                
                    fig = fviz_pca_var(
                            self=model,
                            axis=[int(input.axis1()),int(input.axis2())],
                            title=input.var_title(),
                            color=input.var_text_actif_color(),
                            quanti_sup=quanti_sup,
                            color_sup=color_sup, 
                            text_size=input.var_text_size(),
                            lim_contrib = input.var_lim_contrib(),
                            lim_cos2 = input.var_lim_cos2() 
                            )
                elif input.var_text_color() in ["cos2","contrib"]:
                    fig = fviz_pca_var(
                            self=model,
                            axis=[int(input.axis1()),int(input.axis2())],
                            title=input.var_title(),
                            color=input.var_text_color(),
                            quanti_sup=quanti_sup,
                            text_size=input.var_text_size(),
                            lim_contrib = input.var_lim_contrib(),
                            lim_cos2 = input.var_lim_cos2() 
                            )
                elif input.var_text_color() == "kmeans":
                    kmeans = KMeans(n_clusters=input.var_text_kmeans_nb_clusters(), random_state=np.random.seed(123), n_init="auto").fit(model.var_["coord"])
                    fig = fviz_pca_var(
                            self=model,
                            axis=[int(input.axis1()),int(input.axis2())],
                            title=input.var_title(),
                            color=kmeans,
                            quanti_sup=quanti_sup,
                            text_size=input.var_text_size(),
                            lim_contrib = input.var_lim_contrib(),
                            lim_cos2 = input.var_lim_cos2() 
                            )
                return fig+pn.theme_gray()

            # Variables Factor Map - PCA
            @render.plot(alt="Correlation circle - PCA")
            def fviz_var_plot():
                return plot_var().draw()
            
            #-------------------------------------------------------------------------------------------
            ## Eigenvalue - Scree plot
            #-------------------------------------------------------------------------------------------
            # Reactive Scree plot
            @reactive.Calc
            def plot_eigen():
                return fviz_eig(self=model,choice=input.fviz_eigen_choice(),add_labels=input.fviz_eigen_label(),ggtheme=pn.theme_gray())

            # Render Scree plot
            @render.plot(alt="Scree Plot - PCA")
            def fviz_eigen():
                return plot_eigen().draw()
            
            # Eigen value - DataFrame
            @render.data_frame
            def eigen_table():
                eig = model.eig_.round(4).reset_index().rename(columns={"index":"dimensions"})
                eig.columns = [x.capitalize() for x in eig.columns]
                return DataTable(data=match_datalength(eig,input.eigen_table_len()),filters=input.eigen_table_filter())
            
            #----------------------------------------------------------------------------------------------------
            ## Quantitative variables
            #----------------------------------------------------------------------------------------------------
            # Factor coordinates - Correlation with axis
            @render.data_frame
            def var_coord_table():
                var_coord = model.var_["coord"].round(4).reset_index()
                var_coord.columns = ["Variables", *var_coord.columns[1:]]
                return DataTable(data=match_datalength(data=var_coord,value=input.var_coord_len()),filters=input.var_coord_filter())
            
            # Contributions
            @render.data_frame
            def var_contrib_table():
                var_contrib = model.var_["contrib"].round(4).reset_index()
                var_contrib.columns = ["Variables", *var_contrib.columns[1:]]
                return  DataTable(data=match_datalength(data=var_contrib,value=input.var_contrib_len()),filters=input.var_contrib_filter())
            
            # Add Variables Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.var_contrib_graph_btn)
            def _():
                GraphModalShow(text="var",name="contrib")
            
            @reactive.Calc
            def var_contrib_plot():
                fig = fviz_contrib(self=model,choice="var",axis=input.var_contrib_axis(),top_contrib=int(input.var_contrib_top()),color=input.var_contrib_color(),bar_width=input.var_contrib_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot variables Contributions
            @render.plot(alt="Variables Contributions Map - PCA")
            def fviz_var_contrib():
                return var_contrib_plot().draw()
            
            # Square cosinus
            @render.data_frame
            def var_cos2_table():
                var_cos2 = model.var_["cos2"].round(4).reset_index()
                var_cos2.columns = ["Variables", *var_cos2.columns[1:]]
                return  DataTable(data=match_datalength(data=var_cos2,value=input.var_cos2_len()),filters=input.var_cos2_filter())
            
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.var_cos2_graph_btn)
            def _():
                GraphModalShow(text="var",name="cos2")
            
            @reactive.Calc
            def var_cos2_plot():
                fig = fviz_cos2(self=model,choice="var",axis=input.var_cos2_axis(),top_cos2=int(input.var_cos2_top()),color=input.var_cos2_color(),bar_width=input.var_cos2_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot variables Cos2
            @render.plot(alt="Variables Cosines Map - PCA")
            def fviz_var_cos2():
                return var_cos2_plot().draw()

            #---------------------------------------------------------------------------------
            ## Supplementary quantitative Variables
            #---------------------------------------------------------------------------------
            if hasattr(model,"quanti_sup_"):
                @render.ui
                def quanti_sup_panel():
                    return ui.panel_conditional("input.value_choice == 'quanti_sup_res'",
                                ui.input_radio_buttons(id="quanti_sup_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                                ui.panel_conditional("input.quanti_sup_choice === 'coord'",PanelConditional1(text="quanti_sup",name="coord")),
                                ui.panel_conditional("input.quanti_sup_choice === 'cos2'",PanelConditional1(text="quanti_sup",name="cos2"))
                            )
            
                # Factor coordinates - correlation with factor
                @render.data_frame
                def quanti_sup_coord_table():
                    quanti_sup_coord = model.quanti_sup_["coord"].round(4).reset_index()
                    quanti_sup_coord.columns = ["Variables", *quanti_sup_coord.columns[1:]]
                    return DataTable(data=match_datalength(data=quanti_sup_coord,value=input.quanti_sup_coord_len()),filters=input.quanti_sup_coord_filter())
                
                # Square cosinus
                @render.data_frame
                def quanti_sup_cos2_table():
                    quanti_sup_cos2 = model.quanti_sup_["cos2"].round(4).reset_index()
                    quanti_sup_cos2.columns = ["Variables", *quanti_sup_cos2.columns[1:]]
                    return DataTable(data=match_datalength(data=quanti_sup_cos2,value=input.quanti_sup_cos2_len()),filters=input.quanti_sup_cos2_filter())
            
            #------------------------------------------------------------------------------------------
            # Supplementary qualitatives variables
            #------------------------------------------------------------------------------------------
            if hasattr(model,"quali_sup_"):
                @render.ui
                def quali_sup_panel():
                    return ui.panel_conditional("input.value_choice == 'quali_sup_res'",
                                ui.input_radio_buttons(id="quali_sup_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","cos2":"Cos2 - Qualité de la représentation","vtest":"Value-test","eta2" : "Eta2 - Rapport de corrélation"},selected="coord",width="100%",inline=True),
                                ui.panel_conditional("input.quali_sup_choice === 'coord'",PanelConditional1(text="quali_sup",name="coord")),
                                ui.panel_conditional("input.quali_sup_choice === 'cos2'",PanelConditional1(text="quali_sup",name="cos2")),
                                ui.panel_conditional("input.quali_sup_choice === 'vtest'",PanelConditional1(text="quali_sup",name="vtest")),
                                ui.panel_conditional("input.quali_sup_choice === 'eta2'",PanelConditional1(text="quali_sup",name="eta2"))
                            )
            
                # Factor coordinates
                @render.data_frame
                def quali_sup_coord_table():
                    quali_sup_coord = model.quali_sup_["coord"].round(4).reset_index()
                    quali_sup_coord.columns = ["Categories", *quali_sup_coord.columns[1:]]
                    return  DataTable(data = match_datalength(quali_sup_coord,input.quali_sup_coord_len()),filters=input.quali_sup_coord_filter())
                
                # Square cosinus
                @render.data_frame
                def quali_sup_cos2_table():
                    quali_sup_cos2 = model.quali_sup_["cos2"].round(4).reset_index()
                    quali_sup_cos2.columns = ["Categories", *quali_sup_cos2.columns[1:]]
                    return  DataTable(data = match_datalength(quali_sup_cos2,input.quali_sup_cos2_len()),filters=input.quali_sup_cos2_filter())
                
                # Value - Test
                @render.data_frame
                def quali_sup_vtest_table():
                    quali_sup_vtest = model.quali_sup_["vtest"].round(4).reset_index()
                    quali_sup_vtest.columns = ["Categories", *quali_sup_vtest.columns[1:]]
                    return  DataTable(data = match_datalength(quali_sup_vtest,input.quali_sup_vtest_len()),filters=input.quali_sup_vtest_filter())
                
                # Square correlation ratio
                @render.data_frame
                def quali_sup_eta2_table():
                    quali_sup_eta2 = model.quali_sup_["eta2"].round(4).reset_index()
                    quali_sup_eta2.columns = ["Variables", *quali_sup_eta2.columns[1:]]
                    return  DataTable(data = match_datalength(quali_sup_eta2,input.quali_sup_eta2_len()),filters=input.quali_sup_eta2_filter())
            
            #--------------------------------------------------------------------------------------------------------
            ## Individuals informations
            #--------------------------------------------------------------------------------------------------------
            # Individuals Coordinates
            @render.data_frame
            def ind_coord_table():
                ind_coord = model.ind_["coord"].round(4).reset_index()
                ind_coord.columns = ["Individus", *ind_coord.columns[1:]]
                return DataTable(data = match_datalength(ind_coord,input.ind_coord_len()),filters=input.ind_coord_filter())
            
            # Individuals Contributions
            @render.data_frame
            def ind_contrib_table():
                ind_contrib = model.ind_["contrib"].round(4).reset_index()
                ind_contrib.columns = ["Individus", *ind_contrib.columns[1:]]
                return  DataTable(data=match_datalength(ind_contrib,input.ind_contrib_len()),filters=input.ind_contrib_filter())
            
            # Add indiviuals Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.ind_contrib_graph_btn)
            def _():
                GraphModalShow(text="ind",name="contrib")
            
            @reactive.Calc
            def ind_contrib_plot():
                fig = fviz_contrib(self=model,choice="ind",axis=input.ind_contrib_axis(),top_contrib=int(input.ind_contrib_top()),color = input.ind_contrib_color(),bar_width= input.ind_contrib_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot Individuals Contributions
            @render.plot(alt="Individuals Contributions Map - PCA")
            def fviz_ind_contrib():
                return ind_contrib_plot().draw()
            
            # Square cosinus
            @render.data_frame
            def ind_cos2_table():
                ind_cos2 = model.ind_["cos2"].round(4).reset_index()
                ind_cos2.columns = ["Individus", *ind_cos2.columns[1:]]
                return  DataTable(data = match_datalength(ind_cos2,input.ind_cos2_len()),filters=input.ind_cos2_filter())
            
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.ind_cos2_graph_btn)
            def _():
                GraphModalShow(text="ind",name="cos2")
            
            @reactive.Calc
            def ind_cos2_plot():
                fig = fviz_cos2(self=model,choice="ind",axis=input.ind_cos2_axis(),top_cos2=int(input.ind_cos2_top()),color=input.ind_cos2_color(),bar_width=input.ind_cos2_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot variables Cos2
            @render.plot(alt="Individuals Cosines Map - PCA")
            def fviz_ind_cos2(): 
                return ind_cos2_plot().draw()
            
            #---------------------------------------------------------------------------------------------
            ## Supplementary individuals informations
            #---------------------------------------------------------------------------------------------
            if hasattr(model,"ind_sup_"):
                @render.ui
                def ind_sup_panel():
                    return ui.panel_conditional("input.value_choice == 'ind_sup_res'",
                                ui.input_radio_buttons(id="ind_sup_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                                ui.panel_conditional("input.ind_sup_choice === 'coord'",PanelConditional1(text="ind_sup",name="coord")),
                                ui.panel_conditional("input.ind_sup_choice === 'cos2'",PanelConditional1(text="ind_sup",name="cos2"))
                            )
                
                # Factor coordinates
                @render.data_frame
                def ind_sup_coord_table():
                    ind_sup_coord = model.ind_sup_["coord"].round(4).reset_index()
                    ind_sup_coord.columns = ["Individus", *ind_sup_coord.columns[1:]]
                    return  DataTable(data = match_datalength(ind_sup_coord,input.ind_sup_coord_len()),filters=input.ind_sup_coord_filter())
                
                # Square cosinus
                @render.data_frame
                def ind_sup_cos2_table():
                    ind_sup_cos2 = model.ind_sup_["cos2"].round(4).reset_index()
                    ind_sup_cos2.columns = ["Individus", *ind_sup_cos2.columns[1:]]
                    return  DataTable(data = match_datalength(ind_sup_cos2,input.ind_sup_cos2_len()),filters=input.ind_sup_cos2_filter())
            
            #---------------------------------------------------------------------------------------
            ## Description of axis
            #----------------------------------------------------------------------------------------
            @reactive.Effect
            def _():
                Dimdesc = dimdesc(self=model,axis=None,proba=float(input.dim_desc_pvalue()))[input.dim_desc_axis()]

                @output
                @render.ui
                def dim_desc():
                    if "quanti" in Dimdesc.keys() and "quali" in Dimdesc.keys():
                        return ui.TagList(
                            ui.input_radio_buttons(id="dim_desc_choice",label=ui.h6("Choice"),choices={"quanti" : "Quantitative","quali" : "Qualitative"},selected="quanti",width="100%",inline=True),
                            ui.panel_conditional("input.dim_desc_choice === 'quanti'",PanelConditional1(text="quanti",name="desc")),
                            ui.panel_conditional("input.dim_desc_choice === 'quali'",PanelConditional1(text="quali",name="desc"))
                        )
                    elif "quanti" in Dimdesc.keys() and "quali" not in Dimdesc.keys():
                        return ui.TagList(
                            ui.input_radio_buttons(id="dim_desc_choice",label=ui.h6("Choice"),choices={"quanti" : "Quantitative"},selected="quanti",width="100%",inline=True),
                            ui.panel_conditional("input.dim_desc_choice === 'quanti'",PanelConditional1(text="quanti",name="desc"))
                        )
                    
                if "quanti" in Dimdesc.keys():
                    @render.data_frame
                    def quanti_desc_table():
                        data = Dimdesc["quanti"].round(4).reset_index().rename(columns={"index":"Variables"})
                        return  DataTable(data = match_datalength(data,input.quanti_desc_len()),filters=input.quanti_desc_filter())
                
                if "quali" in Dimdesc.keys():
                    @render.data_frame
                    def quali_desc_table():
                        data = Dimdesc["quali"].round(4).reset_index().rename(columns={"index":"Variables"})
                        return  DataTable(data = match_datalength(data,input.quali_desc_len()),filters=input.quali_desc_filter())
            
            #-----------------------------------------------------------------------------------------------
            ## Summary of data
            #-----------------------------------------------------------------------------------------------
            # Descriptive statistics
            @render.data_frame
            def stats_desc_table():
                data = model.call_["Xtot"].loc[:,var_labels]
                if model.ind_sup is not None:
                    data = data.drop(index=model.call_["ind_sup"])
                stats_desc = data.describe(include="all").round(4).T.reset_index().rename(columns={"index":"Variables"})
                return  DataTable(data = match_datalength(stats_desc,input.stats_desc_len()),filters=input.stats_desc_filter())

            # Histogramme
            @render.plot(alt="Histogram")
            def fviz_hist_plot():
                data = model.call_["Xtot"].loc[:,var_labels]
                if model.ind_sup is not None:
                    data = data.drop(index=model.call_["ind_sup"])

                p = pn.ggplot(data,pn.aes(x=input.var_label()))
                # Add density
                if input.add_density():
                    p = (p + pn.geom_histogram(pn.aes(y="..density.."), color="black", fill="gray")+
                        pn.geom_density(alpha=.2, fill="#FF6666"))
                else:
                    p = p + pn.geom_histogram(color="black", fill="gray")
                
                p = p + pn.ggtitle(f"Histogram de {input.var_label()}")

                return p.draw()

            # Matrice des corrélations
            @render.data_frame
            def corr_matrix_table():
                data = model.call_["Xtot"].loc[:,var_labels]
                if model.ind_sup is not None:
                    data = data.drop(index=model.call_["ind_sup"])
                corr_mat = data.corr(method="pearson").round(4).reset_index().rename(columns={"index":"Variables"})
                return DataTable(data = match_datalength(corr_mat,input.corr_matrix_len()),filters=input.corr_matrix_filter())

            # Bar plot
            if hasattr(model,"quali_sup_"):
                @render.ui
                def quali_sup_graph():
                    return ui.panel_conditional("input.resume_choice === 'bar_plot'",
                            ui.row(
                                ui.column(2,
                                    ui.input_select(id="var_qual_label",label=ui.h6("Choisir une variable"),choices={x:x for x in model.quali_sup_["eta2"].index},selected=model.quali_sup_["eta2"].index[0])
                                ),
                                ui.column(10,
                                    ui.div(ui.output_plot(id="fviz_bar_plot",width='100%',height='500px'),align="center"),
                                    ui.hr(),
                                    ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                                    ui.div(ui.download_button(id="download_bar_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;",align="center"),
                                    ui.div(ui.download_button(id="download_bar_plot_png",label="png",style = download_btn_style),style="display: inline-block;",align="center"),
                                    ui.div(ui.download_button(id="download_bar_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;",align="center"),
                                    align="center"
                                )
                            )
                        )

                # Diagramme en barres
                @render.plot(alt="Bar-Plot")
                def fviz_bar_plot():
                    data = model.call_["Xtot"].loc[:,model.quali_sup_["eta2"].index.tolist()].astype("object")
                    if model.ind_sup is not None:
                        data = data.drop(index=model.call_["ind_sup"])
                    return (pn.ggplot(data,pn.aes(x=input.var_qual_label()))+ pn.geom_bar()).draw()
            
            #--------------------------------------------------------------------------------------------------
            ## Overall Data
            #--------------------------------------------------------------------------------------------------
            @render.data_frame
            def overall_data_table():
                overall_data = model.call_["Xtot"].reset_index().rename(columns={"index":"Individus"})
                return DataTable(data = match_datalength(overall_data,input.overall_data_len()),filters=input.overall_data_filter())
            
        self.app_ui = app_ui
        self.app_server = server
    
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