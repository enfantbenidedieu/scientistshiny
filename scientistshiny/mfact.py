# -*- coding: utf-8 -*-
from shiny import Inputs, Outputs, Session, render, ui, reactive
import shinyswatch
import numpy as np
import plotnine as pn
import matplotlib.colors as mcolors
from pathlib import Path
from sklearn.cluster import KMeans
from scientisttools import fviz_mfa_ind,fviz_mfa_freq,fviz_mfa_group,fviz_mfa_axes,fviz_eig,fviz_contrib,fviz_cos2,dimdesc
from scientistshiny.base import Base
from scientistshiny.function import *


class MFACTshiny(Base):
    """
    Multiple Factor Analysis for Contingency Tables (MFACT) with scientistshiny
    ---------------------------------------------------------------------------

    Description
    -----------
    Performs Multiple Factor Analysis for Contingency Tables (MFACT) with supplementary individuals and supplementary contingency tables on a Shiny for Python application. Allows to change MFACT graphical parmeters. Graphics can be downloaded in png, jpg and pdf.

    Usage
    -----
    ```python
    >>> MFACTshiny(model)
    ```

    Parameters
    ----------
    `model`: an object of class MFACT. A MFACT result from scientisttools.

    Returns
    -------
    `Graphs` : a tab containing the individuals factor map, the frequency factor map, the groups factor map and the axes factor map.

    `Values` : a tab containing the eigenvalue, the results for the frequency, the results for the individuals, the results for the groups, the results for the partiels axis, the results for the supplementary elements (individuals, frequences and groups).

    `Automatic description of axes` : a tab containing the output of the dimdesc function. This function is designed to point out the variables and the categories that are the most characteristic according to each dimension obtained by a Factor Analysis.

    `Summary of dataset` : a tab containing the summary of the dataset : descriptive statistics.

    `Data` : a tab containing the dataset with a nice display.

    The left part of the application allows to change some elements of the graphs (axes, variables, colors,.)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load mortality dataset
    >>> import pandas as pd
    >>> from scientisttools import MFACT, load_mortality
    >>> from scientistshiny import MFACTshiny
    >>> mortality = load_mortality()
    >>> mortality2 = mortality.copy()
    >>> mortality2.columns = [x + "-2" for x in mortality2.columns]
    >>> dat = pd.concat((mortality,mortality2),axis=1)
    >>> res_mfact = MFACT(group=[9]*4,name_group=["1979","2006","1979-2","2006-2"],num_group_sup=[2,3],ind_sup=list(range(50,dat.shape[0])),parallelize=True).fit(dat)
    >>> res_shiny = MFACTshiny(model=res_mfact)
    >>> res_shiny.run()
    ```
    
    for jupyter notebooks
    https://stackoverflow.com/questions/74070505/how-to-run-fastapi-application-inside-jupyter
    """
    def __init__(self,model=None):
        # Check if model is Multiple Factor Analysis for Contingency Table (MFACT)
        if model.model_ != "mfact":
            raise TypeError("'model' must be an object of class MFACT")
    
        # Initialise value choice
        value_choice = {"eigen_res":"Valeurs propres","freq_res":"Résultats pour les frequences","ind_res":"Résultats pour les individus","group_res" : "Resultats pour les groupes","axes_res": "Résultats pour les axes partiels"}

        # Check if supplementary individuals
        if hasattr(model, "ind_sup_"):
            value_choice.update({"ind_sup_res" : "Résultats pour les individus supplémentaires"})
        
        # Check if supplementary quantitatives variables
        if hasattr(model, "freq_sup_"):
            value_choice.update({"freq_sup_res" : "Résultats pour les frequences supplémentaires"})
        
        # Add supplementary group radio buttons
        if model.num_group_sup is not None:
            value_choice.update({"group_sup_res":"Résultats pour les groupes supplémentaires"})
        
        # UI
        app_ui = ui.page_fluid(
            ui.include_css(css_path),
            shinyswatch.theme.superhero(),
            header(title="Analyse Factorielle Multiple pour tableaux binaires",model_name="MFACT"),
            ui.page_sidebar(
                ui.sidebar(
                    ui.panel_well(
                        ui.h6("Options graphiques",style="text-align:center"),
                        ui.div(ui.h6("Axes"),style="display: inline-block;padding: 5px"),
                        axes_input_select(model=model),
                        ui.br(),
                        ui.div(ui.input_select(id="fviz_choice",label="Quel graphe voule-vous modifier?",choices={"fviz_ind":"Individus","fviz_freq":"Fréquences","fviz_group": "Groupes","fviz_axes":"Axes partiels"},selected="fviz_ind",multiple=False,width="100%")),
                        ui.panel_conditional("input.fviz_choice ==='fviz_ind'",
                            title_input(id="ind_title",value="Individuals - MFACT"),
                            text_size_input(which="ind"),
                            point_select_input(id="ind_point_select"),
                            ui.panel_conditional("input.ind_point_select === 'cos2'",ui.div(lim_cos2(id="ind_lim_cos2"),align="center")),
                            ui.panel_conditional("input.ind_point_select === 'contrib'",ui.div(lim_contrib(id="ind_lim_contrib"),align="center")              ),
                            text_color_input(id="ind_text_color",choices={"actif/sup":"actifs/supplémentaires","cos2":"Cosinus","contrib":"Contribution","kmeans":"KMeans"}),
                            ui.panel_conditional("input.ind_text_color === 'actif/sup'",
                                ui.input_select(id="ind_text_actif_color",label="Individus actifs",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                                ui.output_ui("ind_text_sup")
                            ),
                            ui.panel_conditional("input.ind_text_color === 'kmeans'",ui.input_numeric(id="ind_text_kmeans_nb_clusters",label="Choix du nombre de clusters",value=2,min=1,max=model.ind_["coord"].shape[0],step=1,width="100%")),
                            ui.input_switch(id="ind_plot_repel",label="repel",value=True)
                        ),
                        ui.panel_conditional("input.fviz_choice ==='fviz_freq'",
                            title_input(id="freq_title",value="Contingency tables - MFACT"),
                            text_size_input(which="freq"),
                            point_select_input(id="freq_point_select"),
                            ui.panel_conditional("input.freq_point_select === 'cos2'",ui.div(lim_cos2(id="freq_lim_cos2"),align="center")              ),
                            ui.panel_conditional("input.freq_point_select === 'contrib'",ui.div(lim_contrib(id="freq_lim_contrib"),align="center")              ),
                            ui.input_select(id="freq_text_color",label="Colorier les flèches par :",choices={"actif/sup":"actives/supplémentaires","cos2":"Cosinus","contrib":"Contribution","group":"Groupes","kmeans":"KMeans"},selected="group",multiple=False,width="100%"),
                            ui.panel_conditional("input.freq_text_color === 'actif/sup'",
                                ui.input_select(id="freq_text_actif_color",label="Fréquences actives",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                                ui.output_ui("freq_text_sup")
                            ),
                            ui.panel_conditional("input.freq_text_color === 'kmeans'",ui.input_numeric(id="freq_text_kmeans_nb_clusters",label="Choix du nombre de clusters",value=2,min=1,max=model.freq_["coord"].shape[0],step=1,width="100%")),
                        ),
                        ui.panel_conditional("input.fviz_choice === 'fviz_group'",
                            title_input(id="group_title",value="Graphe des groupes - MFACT"),
                            text_size_input(which="group"),
                            ui.input_select(id="group_text_color",label="Colorier les points par :",choices={"actif/sup":"actifs/supplémentaires","cos2":"Cosinus","contrib":"Contribution","kmeans":"KMeans"},selected="actif/sup",multiple=False,width="100%"),
                            ui.panel_conditional("input.group_text_color ==='actif/sup'",
                                ui.input_select(id="group_text_actif_color",label="Groupes actifs",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                                ui.output_ui("group_text_sup")
                            ),
                            ui.panel_conditional("input.group_text_color === 'kmeans'",ui.input_numeric(id="group_text_kmeans_nb_clusters",label="Choix du nombre de clusters",value=2,min=1,max=model.group_["coord"].shape[0],step=1,width="100%")),
                            ui.input_switch(id="group_plot_repel",label="repel",value=True)
                        ),
                        ui.panel_conditional("input.fviz_choice === 'fviz_axes'",
                            title_input(id="axes_title",value="Graphe des axes partiels - MFACT"),
                            text_size_input(which="axes"),
                            ui.input_select(id="axes_text_color",label="Colorier les points par :",choices={"actif/sup":"actifs/supplémentaires","group":"Groupes"},selected="group",multiple=False,width="100%"),
                            ui.panel_conditional("input.axes_text_color ==='actif/sup'",ui.input_select(id="axes_text_actif_color",label="Axes actifs/supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"))
                        )
                    ),
                    ui.div(ui.input_action_button(id="exit",label="Quitter l'application",style='padding:5px; background-color: #2e4053;text-align:center;white-space: normal;'),align="center"),
                    width="25%"
                ),
                ui.navset_card_tab(
                    ui.nav_panel("Graphes",
                        ui.row(
                            ui.column(6,
                                ui.div(ui.output_plot("fviz_ind_plot",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px"),
                                ui.div(ui.download_button(id="download_ind_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_ind_plot_png",label="png",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_ind_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                align="center"
                            ),
                            ui.column(6,
                                ui.div(ui.output_plot("fviz_freq_plot",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                                ui.div(ui.download_button(id="download_freq_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_freq_plot_png",label="png",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_freq_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                align="center"
                            )
                        ),
                        ui.br(),
                        ui.row(
                            ui.column(6,
                                ui.div(ui.output_plot("fviz_group_plot",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px"),
                                ui.div(ui.download_button(id="download_group_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_group_plot_png",label="png",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_group_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                align="center"
                            ),
                            ui.column(6,
                                ui.div(ui.output_plot("fviz_axes_plot",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                                ui.div(ui.download_button(id="download_axes_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_axes_plot_png",label="png",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_axes_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                align="center"
                            )
                        )
                    ),
                    ui.nav_panel("Valeurs",
                        ui.input_radio_buttons(id="value_choice",label=ui.h6("Quelles sorties voulez-vous?"),choices=value_choice,inline=True),
                        ui.br(),
                        eigen_panel(),
                        ui.panel_conditional("input.value_choice === 'freq_res'",
                            ui.input_radio_buttons(id="freq_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","contrib":"Contributions","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                            ui.panel_conditional("input.freq_choice === 'coord'",panel_conditional1(text="freq",name="coord")),
                            ui.panel_conditional("input.freq_choice === 'contrib'",panel_conditional2(text="freq",name="contrib")),
                            ui.panel_conditional("input.freq_choice === 'cos2'",panel_conditional2(text="freq",name="cos2"))
                        ),
                        ui.panel_conditional("input.value_choice === 'ind_res'",
                            ui.input_radio_buttons(id="ind_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","contrib":"Contributions","cos2":"Cos2 - Qualité de la représentation","partiel_inertia":"Inertie partielle","coord_partial":"Partial coordinates","within_partial_inertia":"Within partial inertia"},selected="coord",width="100%",inline=True),
                            ui.panel_conditional("input.ind_choice === 'coord'",panel_conditional1(text="ind",name="coord")),
                            ui.panel_conditional("input.ind_choice === 'contrib'",panel_conditional2(text="ind",name="contrib")),
                            ui.panel_conditional("input.ind_choice === 'cos2'",panel_conditional2(text="ind",name="cos2")),
                            ui.panel_conditional("input.ind_choice === 'partiel_inertia'",panel_conditional1(text="ind",name="partiel_inertia")),
                            ui.panel_conditional("input.ind_choice === 'coord_partial'",panel_conditional1(text="ind",name="coord_partial")),
                            ui.panel_conditional("input.ind_choice === 'within_partial_inertia'",panel_conditional1(text="ind",name="within_partial_inertia"))
                        ),
                        ui.panel_conditional("input.value_choice === 'group_res'",
                            ui.input_radio_buttons(id="group_choice",label=ui.h6("Quel type de résultats?"),choices={"coef_lg":"Coefficients Lg","coef_rv":"Coefficients RV","coord":"Coordonnées","contrib":"Contribution","cos2":"Cos2 - Qualité de la représentation","cor":"Correlation"},selected="coef_lg",width="100%",inline=True),
                            ui.panel_conditional("input.group_choice === 'coef_lg'",panel_conditional1(text="group",name="coef_lg")),
                            ui.panel_conditional("input.group_choice === 'coef_rv'",panel_conditional1(text="group",name="coef_rv")),
                            ui.panel_conditional("input.group_choice === 'coord'",panel_conditional1(text="group",name="coord")),
                            ui.panel_conditional("input.group_choice === 'contrib'",panel_conditional2(text="group",name="contrib")),
                            ui.panel_conditional("input.group_choice === 'cos2'",panel_conditional2(text="group",name="cos2")),
                            ui.panel_conditional("input.group_choice === 'cor'",panel_conditional1(text="group",name="cor"))
                        ),
                        ui.panel_conditional("input.value_choice === 'axes_res'",
                            ui.input_radio_buttons(id="axes_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","cor":"Corrélations","contrib":"Contributions","cor_inter":"Corrélations inter"},selected="coord",width="100%",inline=True),
                            ui.panel_conditional("input.axes_choice === 'coord'",panel_conditional1(text="axes",name="coord")),
                            ui.panel_conditional("input.axes_choice === 'cor'",panel_conditional1(text="axes",name="cor")),
                            ui.panel_conditional("input.axes_choice === 'contrib'",panel_conditional2(text="axes",name="contrib")),
                            ui.panel_conditional("input.axes_choice === 'cor_inter'",panel_conditional1(text="axes",name="cor_inter"))
                        ),
                        ui.output_ui("ind_sup_panel"),
                        ui.output_ui("freq_sup_panel"),
                        ui.output_ui("group_sup_panel")
                    ),
                    dim_desc_panel(model=model),
                    ui.nav_panel("Résumé du jeu de données",panel_conditional1(text="stats",name="desc")),
                    ui.nav_panel("Données",panel_conditional1(text="overall",name="data"))
                ),
                class_="bslib-page-dashboard",
                fillable=True
            )
        )

        # Server
        def server(input:Inputs, output:Outputs, session:Session):

            #----------------------------------------------------------------------------------------------
            # Disable x and y axis
            @reactive.Effect
            def _():
                Dim = [i for i in range(model.call_["n_components"]) if i > int(input.axis1())]
                ui.update_select(id="axis2",label="",choices={x : x for x in Dim},selected=Dim[0])
            
            @reactive.Effect
            def _():
                Dim = [i for i in range(model.call_["n_components"]) if i < int(input.axis2())]
                ui.update_select(id="axis1",label="",choices={x : x for x in Dim},selected=Dim[0])
            
            #--------------------------------------------------------------------------------------------------
            if hasattr(model,"ind_sup_"):
                @render.ui
                def ind_text_sup():
                    return ui.TagList(ui.input_select(id="ind_text_sup_color",label="Individus supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="blue",multiple=False,width="100%"))

            #--------------------------------------------------------------------------------------------------------------
            if hasattr(model,"ind_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_actif_color",label="Individus actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.ind_text_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_sup_color",label="Individus supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.ind_text_actif_color()]},selected="blue")
            
            #-----------------------------------------------------------------------------------------------
            if hasattr(model,"freq_sup_"):
                @render.ui
                def freq_text_sup():
                        return ui.TagList(ui.input_select(id="freq_text_sup_color",label="Fréquences supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="red",multiple=False,width="100%"))
            
                # Disable colors
                @reactive.Effect
                def _():
                    ui.update_select(id="freq_text_actif_color",label="Fréquences actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.freq_text_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="freq_text_sup_color",label="Fréquences supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.freq_text_actif_color()]},selected="blue")

            #-----------------------------------------------------------------------------------------------
            if model.num_group_sup is not None:
                @render.ui
                def group_text_sup():
                    return ui.TagList(ui.input_select(id="group_text_sup_color",label="Groupes supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="blue",multiple=False,width="100%"))
               
                # Disable groups colors
                @reactive.Effect
                def _():
                    ui.update_select(id="group_text_actif_color",label="Groupes actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.group_text_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="group_text_sup_color",label="Groupes supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.group_text_actif_color()]},selected="blue")

            #--------------------------------------------------------------------------------
            ## Individuals - MFA
            #--------------------------------------------------------------------------------
            @reactive.Calc
            def plot_ind():
                # Define boolean
                if hasattr(model,"ind_sup_"):
                    ind_sup = True
                else:
                    ind_sup = False
                
                if input.ind_text_color() == "actif/sup":
                    # Define colors for supplementary individuals
                    if hasattr(model,"ind_sup_"):
                        color_sup = input.ind_text_sup_color()
                    else:
                        color_sup = None
                    
                    fig = fviz_mfa_ind(self=model,
                                       axis=[int(input.axis1()),int(input.axis2())],
                                       color=input.ind_text_actif_color(),
                                       ind_sup=ind_sup,
                                       quali_sup=False,
                                       color_sup = color_sup,
                                       text_size = input.ind_text_size(),
                                       lim_contrib =input.ind_lim_contrib(),
                                       lim_cos2 = input.ind_lim_cos2(),
                                       title = input.ind_title(),
                                       repel=input.ind_plot_repel())
                elif input.ind_text_color() in ["cos2","contrib"]:
                    fig = fviz_mfa_ind(self=model,
                                       axis=[int(input.axis1()),int(input.axis2())],
                                       color=input.ind_text_color(),
                                       ind_sup=ind_sup,
                                       quali_sup=False,
                                       text_size = input.ind_text_size(),
                                       lim_contrib = input.ind_lim_contrib(),
                                       lim_cos2 = input.ind_lim_cos2(),
                                       title = input.ind_title(),
                                       repel=input.ind_plot_repel())
                elif input.ind_text_color() == "kmeans":
                    kmeans = KMeans(n_clusters=input.ind_text_kmeans_nb_clusters(), random_state=np.random.seed(123), n_init="auto").fit(model.ind_["coord"])
                    fig = fviz_mfa_ind(self = model,
                                       axis = [int(input.axis1()),int(input.axis2())],
                                       color = kmeans,
                                       ind_sup = ind_sup,
                                       quali_sup=False,
                                       text_size = input.ind_text_size(),
                                       lim_contrib = input.ind_lim_contrib(),
                                       lim_cos2 = input.ind_lim_cos2(),
                                       title = input.ind_title(),
                                       repel=input.ind_plot_repel())
                return fig+pn.theme_gray()

            @render.plot(alt="Individuals - MFACT")
            def fviz_ind_plot():
                return plot_ind().draw()
            
            # Downlaod
            # @render.download(filename="Individuals-Factor-Map.png")
            # def download_ind_plot_png():
            #    return plot_ind().save("Individuals-Factor-Map.png")
            
            #---------------------------------------------------------------------------------------------
            ##  Contingency tables - MFACT
            #---------------------------------------------------------------------------------------------
            @reactive.Calc
            def plot_freq():
                # Define boolean 
                if hasattr(model, "freq_sup_"):
                    freq_sup = True
                else:
                    freq_sup = False
                
                if input.freq_text_color() == "actif/sup":
                    if hasattr(model, "freq_sup_"):
                        color_sup = input.freq_text_sup_color()
                    else:
                        color_sup = None
                    
                    fig = fviz_mfa_freq(self=model,
                                       axis=[int(input.axis1()),int(input.axis2())],
                                       title=input.freq_title(),
                                       color=input.freq_text_actif_color(),
                                       freq_sup=freq_sup,
                                       color_sup=color_sup,
                                       text_size=input.freq_text_size(),
                                       lim_contrib = input.freq_lim_contrib(),
                                       lim_cos2 = input.freq_lim_cos2())
                elif input.freq_text_color() in ["cos2","contrib","group"]:
                    fig = fviz_mfa_freq(self=model,
                                       axis=[int(input.axis1()),int(input.axis2())],
                                       title=input.freq_title(),
                                       color=input.freq_text_color(),
                                       freq_sup=freq_sup,
                                       text_size=input.freq_text_size(),
                                       lim_contrib = input.freq_lim_contrib(),
                                       lim_cos2 = input.freq_lim_cos2())
                elif input.freq_text_color() == "kmeans":
                    kmeans = KMeans(n_clusters=input.freq_text_kmeans_nb_clusters(), random_state=np.random.seed(123), n_init="auto").fit(model.freq_["coord"])
                    fig = fviz_mfa_freq(self=model,
                                       axis=[int(input.axis1()),int(input.axis2())],
                                       title=input.freq_title(),
                                       color=kmeans,
                                       freq_sup=freq_sup,
                                       text_size=input.freq_text_size(),
                                       lim_contrib = input.freq_lim_contrib(),
                                       lim_cos2 = input.freq_lim_cos2())
                return fig+pn.theme_gray()

            @render.plot(alt="Contingency tables - MFACT")
            def fviz_freq_plot():
                return plot_freq().draw()
            
            #-----------------------------------------------------------------
            # Groups Factor Map
            #-----------------------------------------------------------------
            @reactive.Calc
            def plot_group():
                if model.num_group_sup is not None:
                    group_sup = True
                else:
                    group_sup = False
                
                if input.group_text_color() =='actif/sup':
                    if model.num_group_sup is not None:
                        color_sup = input.group_text_sup_color()
                    else:
                        color_sup = None
                    
                    fig = fviz_mfa_group(self=model,
                                         axis=[int(input.axis1()),int(input.axis2())],
                                         title=input.group_title(),
                                         color=input.group_text_actif_color(),
                                         group_sup=group_sup,
                                         color_sup=color_sup,
                                         text_size=input.group_text_size(),
                                         repel=input.group_plot_repel())
                elif input.group_text_color() in ["cos2","contrib"]:
                    fig = fviz_mfa_group(self=model,
                                         axis=[int(input.axis1()),int(input.axis2())],
                                         title=input.group_title(),
                                         color=input.group_text_color(),
                                         group_sup=group_sup,
                                         text_size=input.group_text_size(),
                                         repel=input.group_plot_repel())
                elif input.group_text_color() == "kmeans":
                    kmeans = KMeans(n_clusters=input.group_text_kmeans_nb_clusters(), random_state=np.random.seed(123), n_init="auto").fit(model.group_["coord"])
                    fig = fviz_mfa_group(self=model,
                                         axis=[int(input.axis1()),int(input.axis2())],
                                         title=input.group_title(),
                                         color=kmeans,
                                         group_sup=group_sup,
                                         text_size=input.group_text_size(),
                                         repel=input.group_plot_repel())
                return fig+pn.theme_gray()
            
            @render.plot(alt="Groups Factor Map - MFACT")
            def fviz_group_plot():
                return plot_group().draw()
            
            #----------------------------------------------------------------------------------
            ## Axes Partiels Factor Map
            #-------------------------------------------------------------------------------------
            @reactive.Calc
            def plot_axes():
                if input.axes_text_color() == "actif/sup":
                    fig = fviz_mfa_axes(self=model,
                                        axis=[int(input.axis1()),int(input.axis2())],
                                        color=input.axes_text_actif_color(),
                                        title=input.axes_title(),
                                        text_size=input.axes_text_size())
                elif input.axes_text_color() == "group":
                    fig = fviz_mfa_axes(self=model,
                                        axis=[int(input.axis1()),int(input.axis2())],
                                        color=input.axes_text_color(),
                                        title=input.axes_title(),
                                        text_size=input.axes_text_size())
                return fig+pn.theme_gray()
            
            @render.plot(alt="Axes partiels Factor Map - MFACT")
            def fviz_axes_plot():
                return plot_axes().draw()
            
            #-------------------------------------------------------------------------------------------
            ## Eigenvalue informations
            #-------------------------------------------------------------------------------------------
            # Eigenvalue - Scree plot
            @render.plot(alt="Scree Plot - MFA")
            def fviz_eigen():
                return fviz_eig(self=model,choice=input.fviz_eigen_choice(),add_labels=input.fviz_eigen_label(),ggtheme=pn.theme_gray()).draw()
            
            # Eigen value - DataFrame
            @render.data_frame
            def eigen_table():
                eig = model.eig_.round(4).reset_index().rename(columns={"index":"dimensions"})
                eig.columns = [x.capitalize() for x in eig.columns]
                return DataTable(data=match_datalength(eig,input.eigen_table_len()),filters=input.eigen_table_filter())
            
            # @render.download(filename=f"MFA-eigenvalue-{date.today().isoformat()}.xlsx")
            # async def eigen_download_xlsx():
            #     yield eigen_table.data_view().to_excel(excel_writer=list[FileInfo][0]["datapath"],index=False)
            
            #----------------------------------------------------------------------------------------------------------------------------------------------
            ## frequences informations
            #----------------------------------------------------------------------------------------------------------------------------------------------
            # Factor Coordinates
            @render.data_frame
            def freq_coord_table():
                freq_coord = model.freq_["coord"].round(4).reset_index()
                freq_coord.columns = ["Frequences", *freq_coord.columns[1:]]
                return DataTable(data=match_datalength(data=freq_coord,value=input.freq_coord_len()),filters=input.freq_coord_filter())
            
            # Contributions
            @render.data_frame
            def freq_contrib_table():
                freq_contrib = model.freq_["contrib"].round(4).reset_index()
                freq_contrib.columns = ["Frequences", *freq_contrib.columns[1:]]
                return  DataTable(data=match_datalength(data=freq_contrib,value=input.freq_contrib_len()),filters=input.freq_contrib_filter())
            
            # Add frequences Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.freq_contrib_graph_btn)
            def _():
                graph_modal_show(text="freq",name="contrib",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def freq_contrib_plot():
                fig = fviz_contrib(self=model,choice="freq",axis=input.freq_contrib_axis(),top_contrib=int(input.freq_contrib_top()),color=input.freq_contrib_color(),bar_width=input.freq_contrib_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot variables Contributions
            @render.plot(alt="Frequences Contributions Map - MFACT")
            def fviz_freq_contrib():
                return freq_contrib_plot().draw()
            
            # Square cosinus 
            @render.data_frame
            def freq_cos2_table():
                freq_cos2 = model.freq_["cos2"].round(4).reset_index()
                freq_cos2.columns = ["Variables", *freq_cos2.columns[1:]]
                return  DataTable(data=match_datalength(data=freq_cos2,value=input.freq_cos2_len()),filters=input.freq_cos2_filter())
            
            # Add frequences  Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.freq_cos2_graph_btn)
            def _():
                graph_modal_show(text="freq",name="cos2",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def freq_cos2_plot():
                fig = fviz_cos2(self=model,choice="freq",axis=input.freq_cos2_axis(),top_cos2=int(input.freq_cos2_top()),color=input.freq_cos2_color(),bar_width=input.freq_cos2_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot variables Cos2
            @render.plot(alt="Frequences cosines Map - MFACT")
            def fviz_freq_cos2():
                return freq_cos2_plot().draw()

            #---------------------------------------------------------------------------------
            ## Supplementary frequences
            #---------------------------------------------------------------------------------
            if hasattr(model,"freq_sup_"):
                @render.ui
                def freq_sup_panel():
                    return ui.panel_conditional("input.value_choice == 'freq_sup_res'",
                                ui.input_radio_buttons(id="freq_sup_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                                ui.panel_conditional("input.freq_sup_choice === 'coord'",panel_conditional1(text="freq_sup",name="coord")),
                                ui.panel_conditional("input.freq_sup_choice === 'cos2'",panel_conditional1(text="freq_sup",name="cos2"))
                            )
                
                # Factor coordinates
                @render.data_frame
                def freq_sup_coord_table():
                    freq_sup_coord = model.freq_sup_["coord"].round(4).reset_index()
                    freq_sup_coord.columns = ["Variables", *freq_sup_coord.columns[1:]]
                    return DataTable(data=match_datalength(data=freq_sup_coord,value=input.freq_sup_coord_len()),filters=input.freq_sup_coord_filter())
                
                # Square cosinus
                @render.data_frame
                def freq_sup_cos2_table():
                    freq_sup_cos2 = model.freq_sup_["cos2"].round(4).reset_index()
                    freq_sup_cos2.columns = ["Variables", *freq_sup_cos2.columns[1:]]
                    return DataTable(data=match_datalength(data=freq_sup_cos2,value=input.freq_sup_cos2_len()),filters=input.freq_sup_cos2_filter())
            
            #--------------------------------------------------------------------------------------------------------
            ## Individuals informations
            #---------------------------------------------------------------------------------------------------------
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
                graph_modal_show(text="ind",name="contrib",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def ind_contrib_plot():
                fig = fviz_contrib(self=model,choice="ind",axis=input.ind_contrib_axis(),top_contrib=int(input.ind_contrib_top()),color = input.ind_contrib_color(),bar_width= input.ind_contrib_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot Individuals Contributions
            @render.plot(alt="Individuals Contributions Map - MFACT")
            def fviz_ind_contrib():
                return ind_contrib_plot().draw()
            
            # Individuals Cos2 
            @render.data_frame
            def ind_cos2_table():
                ind_cos2 = model.ind_["cos2"].round(4).reset_index()
                ind_cos2.columns = ["Individus", *ind_cos2.columns[1:]]
                return  DataTable(data = match_datalength(ind_cos2,input.ind_cos2_len()),filters=input.ind_cos2_filter())
            
            # Add Individuals Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.ind_cos2_graph_btn)
            def _():
                graph_modal_show(text="ind",name="cos2",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def ind_cos2_plot():
                fig = fviz_cos2(self=model,choice="ind",axis=input.ind_cos2_axis(),top_cos2=int(input.ind_cos2_top()),color=input.ind_cos2_color(),bar_width=input.ind_cos2_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot variables Cos2
            @render.plot(alt="Individuals Cosines Map - MFACT")
            def fviz_ind_cos2(): 
                return ind_cos2_plot().draw()
            
            # individuals inertie partielle
            @render.data_frame
            def ind_partiel_inertia_table():
                ind_partiel_inertia = model.ind_["within_inertia"].round(4).reset_index()
                ind_partiel_inertia.columns = ["Individus", *ind_partiel_inertia.columns[1:]]
                return  DataTable(data = match_datalength(ind_partiel_inertia,input.ind_partiel_inertia_len()),filters=input.ind_partiel_inertia_filter())
            
            # individuals partial coordinates
            @render.data_frame
            def ind_coord_partial_table():
                ind_coord_partial = reset_columns(X=model.ind_["coord_partiel"].round(4)).reset_index()
                ind_coord_partial.columns = ["Individus", *ind_coord_partial.columns[1:]]
                return  DataTable(data = match_datalength(ind_coord_partial,input.ind_coord_partial_len()),filters=input.ind_coord_partial_filter())
            
            # Individuals Within partial inertia
            @render.data_frame
            def ind_within_partial_inertia_table():
                ind_within_partial_inertia = reset_columns(X=model.ind_["within_partial_inertia"].round(4)).reset_index()
                ind_within_partial_inertia.columns = ["Individus", *ind_within_partial_inertia.columns[1:]]
                return  DataTable(data = match_datalength(ind_within_partial_inertia,input.ind_within_partial_inertia_len()),filters=input.ind_within_partial_inertia_filter())

            #---------------------------------------------------------------------------------------------
            ## Supplementary individuals informations
            #---------------------------------------------------------------------------------------------
            # Supplementary individuals conditional Panel
            @render.ui
            def ind_sup_panel():
                return ui.panel_conditional("input.value_choice == 'ind_sup_res'",
                            ui.input_radio_buttons(id="ind_sup_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","cos2":"Cos2 - Qualité de la représentation","coord_partiel": "Partial coordinates"},selected="coord",width="100%",inline=True),
                            ui.br(),
                            ui.panel_conditional("input.ind_sup_choice === 'coord'",panel_conditional1(text="ind_sup",name="coord")),
                            ui.panel_conditional("input.ind_sup_choice === 'cos2'",panel_conditional1(text="ind_sup",name="cos2")),
                            ui.panel_conditional("input.ind_sup_choice === 'coord_partiel'",panel_conditional1(text="ind_sup",name="coord_partiel"))
                        )
            
            # Supplementary Individual Coordinates
            @render.data_frame
            def ind_sup_coord_table():
                ind_sup_coord = model.ind_sup_["coord"].round(4).reset_index()
                ind_sup_coord.columns = ["Individus", *ind_sup_coord.columns[1:]]
                return  DataTable(data = match_datalength(ind_sup_coord,input.ind_sup_coord_len()),filters=input.ind_sup_coord_filter())
            
            # Supplementary Individual Cos2
            @render.data_frame
            def ind_sup_cos2_table():
                ind_sup_cos2 = model.ind_sup_["cos2"].round(4).reset_index()
                ind_sup_cos2.columns = ["Individus", *ind_sup_cos2.columns[1:]]
                return  DataTable(data = match_datalength(ind_sup_cos2,input.ind_sup_cos2_len()),filters=input.ind_sup_cos2_filter())
            
            # Supplementary Individual Partiel coordinates
            @render.data_frame
            def ind_sup_coord_partiel_table():
                ind_sup_coord_partiel = reset_columns(X=model.ind_sup_["coord_partiel"].round(4)).reset_index()
                ind_sup_coord_partiel.columns = ["Individus", *ind_sup_coord_partiel.columns[1:]]
                return  DataTable(data = match_datalength(ind_sup_coord_partiel,input.ind_sup_coord_partiel_len()),filters=input.ind_sup_coord_partiel_filter())
            
            #----------------------------------------------------------------------------------
            ## Groups informations
            #----------------------------------------------------------------------------------
            # Groups Lg coefficients
            @render.data_frame
            def group_coef_lg_table():
                group_coef_lg = model.group_["Lg"].round(4).reset_index()
                group_coef_lg.columns = ["group", *group_coef_lg.columns[1:]]
                return  DataTable(data = match_datalength(group_coef_lg,input.group_coef_lg_len()),filters=input.group_coef_lg_filter())
            
            # Groups RV coefficients
            @render.data_frame
            def group_coef_rv_table():
                group_coef_rv = model.group_["RV"].round(4).reset_index()
                group_coef_rv.columns = ["group", *group_coef_rv.columns[1:]]
                return  DataTable(data = match_datalength(group_coef_rv,input.group_coef_rv_len()),filters=input.group_coef_rv_filter())
            
            # Groups coordinates
            @render.data_frame
            def group_coord_table():
                group_coord = model.group_["coord"].round(4).reset_index()
                group_coord.columns = ["group", *group_coord.columns[1:]]
                return  DataTable(data = match_datalength(group_coord,input.group_coord_len()),filters=input.group_coord_filter())
            
            # Groups contributions
            @render.data_frame
            def group_contrib_table():
                group_contrib = model.group_["contrib"].round(4).reset_index()
                group_contrib.columns = ["group", *group_contrib.columns[1:]]
                return  DataTable(data = match_datalength(group_contrib,input.group_contrib_len()),filters=input.group_contrib_filter())
            
            # Add group Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.group_contrib_graph_btn)
            def _():
                graph_modal_show(text="group",name="contrib",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def group_contrib_plot():
                fig = fviz_contrib(self=model,choice="group",axis=input.group_contrib_axis(),top_contrib=int(input.group_contrib_top()),color = input.group_contrib_color(),bar_width= input.group_contrib_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot Individuals Contributions
            @render.plot(alt="Group Contributions Map - MFACT")
            def fviz_group_contrib():
                return group_contrib_plot().draw()
            
            # Groups Square cosinus
            @render.data_frame
            def group_cos2_table():
                group_cos2 = model.group_["coord"].round(4).reset_index()
                group_cos2.columns = ["group", *group_cos2.columns[1:]]
                return  DataTable(data = match_datalength(group_cos2,input.group_cos2_len()),filters=input.group_cos2_filter())
            
            # Add Group Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.group_cos2_graph_btn)
            def _():
                graph_modal_show(text="group",name="cos2",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def group_cos2_plot():
                fig = fviz_cos2(self=model,choice="group",axis=input.group_cos2_axis(),top_cos2=int(input.group_cos2_top()),color=input.group_cos2_color(),bar_width=input.group_cos2_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot Group cos2
            @render.plot(alt="Group Cosines Map - MFACT")
            def fviz_group_cos2(): 
                return group_cos2_plot().draw()
            
            # Groups correlations
            @render.data_frame
            def group_cor_table():
                group_cor = model.group_["correlation"].round(4).reset_index()
                group_cor.columns = ["group", *group_cor.columns[1:]]
                return  DataTable(data = match_datalength(group_cor,input.group_cor_len()),filters=input.group_cor_filter())
            
            #--------------------------------------------------------------------------------------------------------
            ## Supplementary group informations
            #--------------------------------------------------------------------------------------------------------
            # Supplementary group conditional panel
            if model.num_group_sup is not None:
                @render.ui
                def group_sup_panel():
                    return ui.panel_conditional("input.value_choice == 'group_sup_res'",
                            ui.input_radio_buttons(id="group_sup_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                            ui.panel_conditional("input.group_sup_choice === 'coord'",panel_conditional1(text="group_sup",name="coord")),
                            ui.panel_conditional("input.group_sup_choice === 'cos2'",panel_conditional1(text="group_sup",name="cos2"))
                        )
                
                # Factor coordinates
                @render.data_frame
                def group_sup_coord_table():
                    group_sup_coord = model.group_["coord_sup"].round(4).reset_index()
                    group_sup_coord.columns = ["Variables", *group_sup_coord.columns[1:]]
                    return DataTable(data=match_datalength(data=group_sup_coord,value=input.group_sup_coord_len()),filters=input.group_sup_coord_filter())
                
                # Square cosinus
                @render.data_frame
                def group_sup_cos2_table():
                    group_sup_cos2 = model.group_["cos2_sup"].round(4).reset_index()
                    group_sup_cos2.columns = ["Variables", *group_sup_cos2.columns[1:]]
                    return DataTable(data=match_datalength(data=group_sup_cos2,value=input.group_sup_cos2_len()),filters=input.group_sup_cos2_filter())
            
            #-------------------------------------------------------------------------------------------
            ## Axes partiels informations
            #-------------------------------------------------------------------------------------------
            # Axes partiel coordinates
            @render.data_frame
            def axes_coord_table():
                axes_coord = model.partial_axes_["coord"].T.round(4).reset_index()
                axes_coord.columns = ["Group","Dimensions", *axes_coord.columns[2:]]
                return  DataTable(data = match_datalength(axes_coord,input.axes_coord_len()),filters=input.axes_coord_filter())
            
            # Axes partiel correlations
            @render.data_frame
            def axes_cor_table():
                axes_cor = model.partial_axes_["cor"].T.round(4).reset_index()
                axes_cor.columns = ["Group","Dimensions", *axes_cor.columns[2:]]
                return  DataTable(data = match_datalength(axes_cor,input.axes_cor_len()),filters=input.axes_cor_filter())
            
            # Axes partiel contrib
            @render.data_frame
            def axes_contrib_table():
                axes_contrib = model.partial_axes_["contrib"].T.round(4).reset_index()
                axes_contrib.columns = ["Group","Dimensions", *axes_contrib.columns[2:]]
                return  DataTable(data = match_datalength(axes_contrib,input.axes_contrib_len()),filters=input.axes_contrib_filter())
            
            # Add group Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.axes_contrib_graph_btn)
            def _():
                graph_modal_show(text="axes",name="contrib",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def axes_contrib_plot():
                fig = fviz_contrib(self=model,choice="partial_axes",axis=input.axes_contrib_axis(),top_contrib=int(input.axes_contrib_top()),color = input.axes_contrib_color(),bar_width= input.axes_contrib_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot Individuals Contributions
            @render.plot(alt="Axes partiels Contributions Map - MFACT")
            def fviz_axes_contrib():
                return axes_contrib_plot().draw()
            
            # Axes partiel correlation inter (correlation between)
            @render.data_frame
            def axes_cor_inter_table():
                axes_cor_inter = reset_columns(X=model.partial_axes_["cor_between"].round(4)).reset_index()
                axes_cor_inter.columns = ["Group","Dimensions", *axes_cor_inter.columns[2:]]
                return  DataTable(data = match_datalength(axes_cor_inter,input.axes_cor_inter_len()),filters=input.axes_cor_inter_filter())

            #------------------------------------------------------------------------------------------------------------
            ## Description of axis
            #------------------------------------------------------------------------------------------------------------
            @reactive.Effect
            def _():
                Dimdesc = dimdesc(self=model,axis=None,proba=float(input.dim_desc_pvalue()))[input.dim_desc_axis()]

                @render.ui
                def dim_desc():
                    if "quanti" in Dimdesc.keys() and "quali" in Dimdesc.keys():
                        return ui.TagList(
                            ui.input_radio_buttons(id="dim_desc_choice",label=ui.h6("Choice"),choices={"quanti" : "Quantitative","quali" : "Qualitative"},selected="quanti",width="100%",inline=True),
                            ui.panel_conditional("input.dim_desc_choice === 'quanti'",panel_conditional1(text="quanti",name="desc")),
                            ui.panel_conditional("input.dim_desc_choice === 'quali'",panel_conditional1(text="quali",name="desc"))
                        )
                    elif "quanti" in Dimdesc.keys() and "quali" not in Dimdesc.keys():
                        return ui.TagList(
                            ui.input_radio_buttons(id="dim_desc_choice",label=ui.h6("Choice"),choices={"quanti" : "Quantitative"},selected="quanti",width="100%",inline=True),
                            ui.panel_conditional("input.dim_desc_choice === 'quanti'",panel_conditional1(text="quanti",name="desc"))
                        )
                    elif "quali" in Dimdesc.keys() and "quanti" not in Dimdesc.keys():
                        return ui.TagList(
                            ui.input_radio_buttons(id="dim_desc_choice",label=ui.h6("Choice"),choices={"quali" : "Qualitative"},selected="quali",width="100%",inline=True),
                            ui.panel_conditional("input.dim_desc_choice === 'quali'",panel_conditional1(text="quali",name="desc"))
                        )
                    else:
                        return ui.TagList(ui.p("No significant variable"))
                    
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
            #--------------------------------------------------------------------------------------------------
            # Descriptive statistics
            @render.data_frame
            def stats_desc_table():
                stats_desc = model.call_["Xtot"].describe(include="all").round(4).T.reset_index().rename(columns={"index":"Variables"})
                return  DataTable(data = match_datalength(stats_desc,input.stats_desc_len()),filters=input.stats_desc_filter())

            #-----------------------------------------------------------------------------------------------------------------------
            # Overall Data
            #-----------------------------------------------------------------------------------------------------------------------
            # Overall Data
            @render.data_frame
            def overall_data_table():
                overall_data = model.call_["Xtot"].reset_index().rename(columns={"index":"Individus"})
                return DataTable(data = match_datalength(overall_data,input.overall_data_len()),filters=input.overall_data_filter())
            
            #-----------------------------------------------------------------------------------------------------------------------
            ## Close the session
            #------------------------------------------------------------------------------------------------------------------------
            @reactive.Effect
            @reactive.event(input.exit)
            async def _():
                await session.close()
            
        self.app_ui = app_ui
        self.app_server = server