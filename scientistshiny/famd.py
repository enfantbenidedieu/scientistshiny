# -*- coding: utf-8 -*-
from shiny import Inputs, Outputs, Session, render, ui, reactive
import shinyswatch
import numpy as np
import pandas as pd
import scipy as sp
import plotnine as pn
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from scientisttools import FAMD,fviz_famd_ind, fviz_famd_mod,fviz_famd_var,fviz_famd_col,fviz_eig, fviz_contrib,fviz_cos2,dimdesc
from scientistshiny.base import Base
from scientistshiny.function import *

class FAMDshiny(Base):
    """
    Factor Analysis of Mixed Data (FAMD) with scientistshiny
    --------------------------------------------------------

    Description
    -----------
    Performs Factor Analysis of Mixed Data (FAMD) with supplementary individuals, supplementary quantitative and/or qualitative variables on a Shiny for Python application. Allows to change FAMD graphical parameters. Graphics can be downloaded in png, jpg and pdf.

    Usage
    -----
    ```python
    >>> FAMDshiny(model)
    ```
    
    Parameters
    ----------
    `model` : a pandas dataframe with n rows (individuals) and p columns (variables) or an instance of class FAMD. A FAMD result from scientisttools.

    Returns
    -------
    `Graphs` : a tab containing the individuals factor map, the correlation circle, the variables categories factor map and the variables factor (quantitative and qualitative)

    `Values` : a tab containing the eigenvalue, the results for the quantitative variables, the results for the qualitatives variables, the  results for variables, the results for the individuals, the results for the supplementary elements (individuals and/or variables).

    `Automatic description of axes` : a tab containing the output of the dimdesc function. This function is designed to point out the variables and the categories that are the most characteristic according to each dimension obtained by a Factor Analysis.

    `Summary of dataset` : a tab containing the summary of the dataset :
        - Pearson correlation matrix and histogram for quantitative variables
        - Bar plot, chi square and others association test for qualitatives variables

    `Data` : a tab containing the dataset with a nice display.

    The left part of the application allows to change some elements of the graphs (axes, variables, colors,.)

    Authors
    -------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load dataset and functions
    >>> from scientisttools import FAMD, load_autos
    >>> from scientistshiny import FAMDshiny
    >>> autos = load_autos()
    >>> # FAMD with scientistshiny
    >>> res_shiny = FAMDshiny(model = autos)
    >>> # FAMDshiny on a results of a FAMD
    >>> res_famd = FAMD(ind_sup=list(range(35,40)),quanti_sup=[10,11],quali_sup=14,parallelize=False).fit(autos)
    >>> res_shiny = FAMDshiny(model = res_famd)
    >>> res_shiny.run()
    ```
    
    for jupyter notebooks
    https://stackoverflow.com/questions/74070505/how-to-run-fastapi-application-inside-jupyter
    """
    def __init__(self,model=None):
        # Check if model is an instance of pd.DataFrame class
        if isinstance(model,pd.DataFrame):        
            # Check if qualitative data
            is_quali = model.select_dtypes(exclude=np.number)
            if is_quali.shape[1]>0:
                for col in is_quali.columns.tolist():
                    model[col] = model[col].astype("object")
            
            # Fit the FAMD with scientisttools
            model = FAMD().fit(model)
        # Check if model FAMD
        if model.model_ != "famd":
            raise ValueError("'model' must be an object of class FAMD")
        
        # Initialise value choice
        value_choice = {"eigen_res":"Valeurs propres","quanti_var_res":"Résultats sur les variables quantitatives","quali_var_res":"Résultats sur les variables qualitatives","var_res":"Résultats sur les variables","ind_res":"Résultats sur les individus"}

        # Quantitative variables labels
        quanti_var_labels = model.quanti_var_["coord"].index.tolist()

        # Qualitative variables labels
        quali_var_labels = [x for x in model.var_["coord"].index if x  not in quanti_var_labels]
        
        if hasattr(model,"ind_sup_"):
            value_choice = {**value_choice,**{"ind_sup_res" : "Résultats des individus supplémentaires"}}
            
        # Check if supplementary quantitatives variables
        if hasattr(model,"quanti_sup_"):
            value_choice = {**value_choice, **{"quanti_sup_res" : "Résultats des variables quantitatives supplémentaires"}}
            quanti_var_labels = [*quanti_var_labels,*model.quanti_sup_["coord"].index.tolist()]

        # Check if supplementary qualitatives variables
        if hasattr(model,"quali_sup_"):
            value_choice = {**value_choice,**{"quali_sup_res" : "Résultats des variables qualitatives supplémentaires"}}
            quali_var_labels = [*quali_var_labels,*model.quali_sup_["eta2"].index.tolist()]
        
        # UI
        app_ui = ui.page_fluid(
            ui.include_css(css_path),
            shinyswatch.theme.superhero(),
            header(title="Analyse Factorielle des données mixtes",model_name="FAMD"),
            ui.page_sidebar(
                ui.sidebar(
                    ui.panel_well(
                        ui.h6("Options graphiques",style="text-align:center"),
                        ui.div(ui.h6("Axes"),style="display: inline-block;padding: 5px"),
                        axes_input_select(model=model),
                        ui.br(),
                        ui.div(ui.input_select(id="fviz_choice",label="Quel graphe voule-vous modifier?",choices={"fviz_ind":"Individus","fviz_var_quant":"Variables quantitatives","fviz_var_qual":"Variables qualitatives","fviz_var": "Variables"},selected="fviz_ind",multiple=False,width="100%")),
                        ui.panel_conditional("input.fviz_choice === 'fviz_ind'",
                            title_input(id="ind_title",value="Individuals - FAMD"),
                            text_size_input(which="ind"),
                            point_select_input(id="ind_point_select"),
                            ui.panel_conditional("input.ind_point_select === 'cos2'",ui.div(lim_cos2(id="ind_lim_cos2"),align="center")),
                            ui.panel_conditional("input.ind_point_select === 'contrib'",ui.div(lim_contrib(id="ind_lim_contrib"),align="center")),
                            text_color_input(id="ind_text_color",choices={"actif/sup":"actifs/supplémentaires","cos2":"Cosinus","contrib":"Contribution","var_quant":"Variable quantitative","var_qual":"Variable qualitative","kmeans" : "KMeans"}),
                            ui.panel_conditional("input.ind_text_color === 'actif/sup'",
                                ui.input_select(id="ind_text_actif_color",label="Individus actifs",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                                ui.input_select(id="ind_text_quali_actif_color",label="Modalités actives",choices={x:x for x in mcolors.CSS4_COLORS},selected="green",multiple=False,width="100%"),
                                ui.output_ui("ind_text_sup"),
                                ui.output_ui("ind_text_quali_sup")
                            ),
                            ui.panel_conditional("input.ind_text_color === 'var_qual'",
                                ui.input_select(id="ind_text_var_qual_color",label="Choix de la variable",choices={x:x for x in quali_var_labels},selected=quali_var_labels[0],multiple=False,width="100%"),
                                ui.input_switch(id="ind_text_add_ellipse",label="Trace les ellipses de confiance autour des barycentres",value=False)
                            ),
                            ui.panel_conditional("input.ind_text_color === 'var_quant'",ui.input_select(id="ind_text_var_quant_color",label="Choix de la variable",choices={x:x for x in quanti_var_labels},selected=quanti_var_labels[0],multiple=False,width="100%")),
                            ui.panel_conditional("input.ind_text_color === 'kmeans'",ui.input_numeric(id="ind_text_kmeans_nb_clusters",label="Choix du nombre de clusters",value=2,min=1,max=model.ind_["coord"].shape[0],step=1,width="100%")),
                            ui.input_switch(id="ind_plot_repel",label="repel",value=True)
                        ),
                        ui.panel_conditional("input.fviz_choice === 'fviz_var_quant'",
                            title_input(id="quanti_var_title",value="Correlation circle - FAMD"),
                            text_size_input(which="quanti_var"),
                            point_select_input(id="quanti_var_point_select"),
                            ui.panel_conditional("input.quanti_var_point_select === 'cos2'",ui.div(lim_cos2(id="quanti_var_lim_cos2"),align="center")),
                            ui.panel_conditional("input.quanti_var_point_select === 'contrib'",ui.div(lim_contrib(id="quanti_var_lim_contrib"),align="center")),
                            text_color_input(id="quanti_var_text_color",choices={"actif/sup": "actifs/supplémentaires","cos2":"Cosinus","contrib":"Contribution","kmeans":"KMeans"}),
                            ui.panel_conditional("input.quanti_var_text_color === 'actif/sup'",
                                ui.input_select(id="quanti_var_text_actif_color",label="Variables quantitatives actives",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                                ui.output_ui("quanti_var_text_sup"),
                            ),
                            ui.panel_conditional("input.quanti_var_text_color === 'kmeans'",ui.input_numeric(id="quanti_var_text_kmeans_nb_clusters",label="Choix du nombre de clusters",value=2,min=1,max=model.quanti_var_["coord"].shape[0],step=1,width="100%")),
                        ),
                        ui.panel_conditional("input.fviz_choice === 'fviz_var_qual'",
                            title_input(id="quali_var_title",value="Variables categories - FAMD"),
                            text_size_input(which="quali_var"),
                            point_select_input(id="quali_var_point_select"),
                            ui.panel_conditional("input.quali_var_point_select === 'cos2'",ui.div(lim_cos2(id="quali_var_lim_cos2"),align="center")),
                            ui.panel_conditional("input.quali_var_point_select === 'contrib'",ui.div(lim_contrib(id="quali_var_lim_contrib"),align="center")),
                            text_color_input(id="quali_var_text_color",choices={"actif/sup": "actifs/supplémentaires","cos2":"Cosinus","contrib":"Contribution","kmeans":"KMeans"}),
                            ui.panel_conditional("input.quali_var_text_color === 'actif/sup'",
                                ui.input_select(id="quali_var_text_actif_color",label="Modalités actives",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                                ui.output_ui("quali_var_text_sup"),
                            ),
                            ui.panel_conditional("input.quali_var_text_color === 'kmeans'",ui.input_numeric(id="quali_var_text_kmeans_nb_clusters",label="Choix du nombre de clusters",value=2,min=1,max=model.quali_var_["coord"].shape[0],step=1,width="100%")),
                            ui.input_switch(id="quali_var_plot_repel",label="repel",value=True)
                        ),
                        ui.panel_conditional("input.fviz_choice === 'fviz_var'",
                            title_input(id="var_title",value="Variables - FAMD"),
                            text_size_input(which="var"),
                            ui.input_select(id="var_quant_text_actif_color",label="Variables quantitatives actives",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                            ui.input_select(id="var_qual_text_actif_color",label="Variables qualitatives actives",choices={x:x for x in mcolors.CSS4_COLORS},selected="green",multiple=False,width="100%"),
                            ui.output_ui("var_quant_text_sup"),
                            ui.output_ui("var_qual_text_sup"),
                            ui.input_switch(id="var_plot_repel",label="repel",value=True)
                        ),
                        ui.div(ui.input_action_button(id="exit",label="Quitter l'application",style='padding:5px; background-color: #2e4053;text-align:center;white-space: normal;'),align="center")
                    ),
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
                                ui.div(ui.output_plot("fviz_quanti_var_plot",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                                ui.div(ui.download_button(id="download_quanti_var_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;",align="center"),
                                ui.div(ui.download_button(id="download_quanti_var_plot_png",label="png",style = download_btn_style),style="display: inline-block;",align="center"),
                                ui.div(ui.download_button(id="download_quanti_var_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;",align="center"),
                                align="center"
                            )
                        ),
                        ui.br(),
                        ui.row(
                            ui.column(6,
                                ui.div(ui.output_plot("fviz_quali_var_plot",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px"),
                                ui.div(ui.download_button(id="download_quali_var_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_quali_var_plot_png",label="png",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_quali_var_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                align="center"
                            ),
                            ui.column(6,
                                ui.div(ui.output_plot("fviz_var_plot",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                                ui.div(ui.download_button(id="download_var_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_var_plot_png",label="png",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_var_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                align="center"
                            )
                        ),
                    ),
                    ui.nav_panel("Valeurs",
                        ui.input_radio_buttons(id="value_choice",label=ui.h6("Quelles sorties voulez-vous?"),choices=value_choice,inline=True),
                        ui.br(),
                        eigen_panel(),
                        ui.panel_conditional("input.value_choice === 'quanti_var_res'",
                            ui.input_radio_buttons(id="quanti_var_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","contrib":"Contributions","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                            ui.panel_conditional("input.quanti_var_choice === 'coord'",panel_conditional1(text="quanti_var",name="coord")),
                            ui.panel_conditional("input.quanti_var_choice === 'contrib'",panel_conditional2(text="quanti_var",name="contrib")),
                            ui.panel_conditional("input.quanti_var_choice === 'cos2'",panel_conditional2(text="quanti_var",name="cos2"))
                        ),
                        ui.panel_conditional("input.value_choice === 'quali_var_res'",
                            ui.input_radio_buttons(id="mod_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","contrib":"Contributions","cos2":"Cos2 - Qualité de la représentation","vtest":"Value - test"},selected="coord",width="100%",inline=True),
                            ui.panel_conditional("input.mod_choice === 'coord'",panel_conditional1(text="quali_var",name="coord")),
                            ui.panel_conditional("input.mod_choice === 'contrib'",panel_conditional2(text="quali_var",name="contrib")),
                            ui.panel_conditional("input.mod_choice === 'cos2'",panel_conditional2(text="quali_var",name="cos2")),
                            ui.panel_conditional("input.mod_choice === 'vtest'",panel_conditional1(text="quali_var",name="vtest"))
                        ),
                        ui.panel_conditional("input.value_choice === 'var_res'",
                            ui.input_radio_buttons(id="var_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","contrib":"Contributions","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                            ui.panel_conditional("input.var_choice === 'coord'",panel_conditional1(text="var",name="coord")),
                            ui.panel_conditional("input.var_choice === 'contrib'",panel_conditional1(text="var",name="contrib")),
                            ui.panel_conditional("input.var_choice === 'cos2'",panel_conditional1(text="var",name="cos2"))
                        ),
                        ui.panel_conditional("input.value_choice === 'ind_res'",
                            ui.input_radio_buttons(id="ind_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","contrib":"Contributions","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                            ui.panel_conditional("input.ind_choice === 'coord'",panel_conditional1(text="ind",name="coord")),
                            ui.panel_conditional("input.ind_choice === 'contrib'",panel_conditional2(text="ind",name="contrib")),
                            ui.panel_conditional("input.ind_choice === 'cos2'",panel_conditional2(text="ind",name="cos2"))
                        ),
                        ui.output_ui("ind_sup_panel"),
                        ui.output_ui("quanti_sup_panel"),
                        ui.output_ui("quali_sup_panel")
                    ),
                    dim_desc_panel(model=model),
                    ui.nav_panel("Résumé du jeu de données",
                        ui.input_radio_buttons(id="resume_choice",label=ui.h6("Quelles sorties voulez - vous?"),choices={"stats_desc":"Statistiques descriptives","hist_plot" : "Histogramme","corr_matrix": "Matrice des corrélations","bar_plot":"Diagramme en barres","chi2_test" : "Test de Chi2","others_test":"Autres mesures d'association"},selected="stats_desc",width="100%",inline=True),
                        ui.br(),
                        ui.panel_conditional("input.resume_choice === 'stats_desc'",panel_conditional1(text="stats",name="desc")),
                        ui.panel_conditional("input.resume_choice === 'hist_plot'",
                            ui.row(
                                ui.column(2,
                                    ui.input_select(id="quanti_var_label",label="Choisir une variable",choices={x:x for x in quanti_var_labels},selected=quanti_var_labels[0],width="100%"),
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
                        ui.panel_conditional("input.resume_choice === 'corr_matrix'",panel_conditional1(text="corr",name="matrix")),
                        ui.panel_conditional("input.resume_choice === 'bar_plot'",
                            ui.row(
                                ui.column(2,ui.input_select(id="quali_var_label",label="Choisir une variable",choices={x:x for x in quali_var_labels},selected=quali_var_labels[0],width="100%")),
                                ui.column(10,
                                    ui.div(ui.output_plot("fviz_bar_plot",width='100%', height='500px'),align="center"),
                                    ui.hr(),
                                    ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px"),
                                    ui.div(ui.download_button(id="download_bar_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                    ui.div(ui.download_button(id="download_bar_plot_png",label="png",style = download_btn_style),style="display: inline-block;"),
                                    ui.div(ui.download_button(id="download_bar_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                    align="center"
                                )
                            )
                        ),
                        ui.panel_conditional("input.resume_choice === 'chi2_test'",panel_conditional1(text="chi2",name="test")),
                        ui.panel_conditional("input.resume_choice === 'others_test'",panel_conditional1(text="others",name="test"))
                    ),
                    ui.nav_panel("Données",panel_conditional1(text="overall",name="data"))
                )
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
            
            if hasattr(model,"quali_sup_"):
                @render.ui
                def ind_text_quali_sup():
                    return ui.TagList(ui.input_select(id="ind_text_quali_sup_color",label="Modalités supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="red",multiple=False,width="100%"))
            
            #-----------------------------------------------------------------------------------------------------
            # Disable individuals colors
            if hasattr(model,"ind_sup_") and hasattr(model,"quali_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_actif_color",label="Individus actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.ind_text_quali_actif_color(),input.ind_text_sup_color(),input.ind_text_quali_sup_color()]]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_quali_actif_color",label="Modalités actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.ind_text_actif_color(),input.ind_text_sup_color(),input.ind_text_quali_sup_color()]]},selected="green")
            
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_sup_color",label="Individus supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.ind_text_actif_color(),input.ind_text_quali_actif_color(),input.ind_text_quali_sup_color()]]},selected="blue")

                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_quali_sup_color",label="Modalités supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.ind_text_actif_color(),input.ind_text_quali_actif_color(),input.ind_text_sup_color()]]},selected="red")
            elif hasattr(model,"ind_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_actif_color",label="Individus actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.ind_text_quali_actif_color(),input.ind_text_sup_color()]]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_quali_actif_color",label="Modalités actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.ind_text_actif_color(),input.ind_text_sup_color()]]},selected="green")
            
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_sup_color",label="Individus supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.ind_text_actif_color(),input.ind_text_quali_actif_color()]]},selected="blue")
            elif hasattr(model,"quali_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_actif_color",label="Individus actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.ind_text_quali_actif_color(),input.ind_text_quali_sup_color()]]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_quali_actif_color",label="Modalités actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.ind_text_actif_color(),input.ind_text_quali_sup_color()]]},selected="green")

                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_quali_sup_color",label="Modalités supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.ind_text_actif_color(),input.ind_text_quali_actif_color()]]},selected="red")
            else:
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_actif_color",label="Individus actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.ind_text_quali_actif_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_quali_actif_color",label="Modalités actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.ind_text_actif_color()]},selected="green")
            
            #-------------------------------------------------------------------------------------------
            if hasattr(model,"quanti_sup_"):
                @render.ui
                def quanti_var_text_sup():
                    return ui.TagList(ui.input_select(id="quanti_var_text_sup_color",label="Variables quantitatives supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="blue",multiple=False,width="100%"))
                
                # Disable quantitative variables colors
                @reactive.Effect
                def _():
                    ui.update_select(id="quanti_var_text_actif_color",label="Variables quantitatives actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.quanti_var_text_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="quanti_var_text_sup_color",label="Variables quantitatives supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.quanti_var_text_actif_color()]},selected="blue")

            #-------------------------------------------------------------------------------------------
            if hasattr(model,"quali_sup_"):
                @render.ui
                def quali_var_text_sup():
                    return ui.TagList(ui.input_select(id="quali_var_text_sup_color",label="Modalités supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="blue",multiple=False,width="100%"))
                
                # Disable qualitative variables colors
                @reactive.Effect
                def _():
                    ui.update_select(id="quali_var_text_actif_color",label="Modalités actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.quali_var_text_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="quali_var_text_sup_color",label="Modalités supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.quali_var_text_actif_color()]},selected="blue")
            
            #------------------------------------------------------------------------------------------
            if hasattr(model,"quanti_sup_"):
                @render.ui
                def var_quant_text_sup():
                    return ui.TagList(ui.input_select(id="var_quant_text_sup_color",label="Variables quantitatives supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="blue",multiple=False,width="100%"))

            if hasattr(model,"quali_sup_"):
                @render.ui
                def var_qual_text_sup():
                    return ui.TagList(ui.input_select(id="var_qual_text_sup_color",label="Variables qualitatives supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="red",multiple=False,width="100%"))

            #----------------------------------------------------------------------------------------------------
            # Disable individuals colors
            if hasattr(model,"quanti_sup_") and hasattr(model,"quali_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="var_quant_text_actif_color",label="Variables quantitatives actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.var_qual_text_actif_color(),input.var_quant_text_sup_color(),input.var_qual_text_sup_color()]]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="var_qual_text_actif_color",label="Variables qualitatives actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.var_quant_text_actif_color(),input.var_quant_text_sup_color(),input.var_qual_text_sup_color()]]},selected="green")
            
                @reactive.Effect
                def _():
                    ui.update_select(id="var_quant_text_sup_color",label="Variables quantitatives supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.var_quant_text_actif_color(),input.var_qual_text_actif_color(),input.var_qual_text_sup_color()]]},selected="blue")

                @reactive.Effect
                def _():
                    ui.update_select(id="var_qual_text_sup_color",label="Variables qualitatives supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.var_quant_text_actif_color(),input.var_qual_text_actif_color(),input.var_quant_text_sup_color()]]},selected="red")
            elif hasattr(model,"quanti_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="var_quant_text_actif_color",label="Variables quantitatives actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.var_qual_text_actif_color(),input.var_quant_text_sup_color()]]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="var_qual_text_actif_color",label="Variables qualitatives actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.var_quant_text_actif_color(),input.var_quant_text_sup_color()]]},selected="green")
            
                @reactive.Effect
                def _():
                    ui.update_select(id="var_quant_text_sup_color",label="Variables quantitatives supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.var_quant_text_actif_color(),input.var_qual_text_actif_color()]]},selected="blue")
            elif hasattr(model,"quali_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="var_quant_text_actif_color",label="Variables quantitatives actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.var_qual_text_actif_color(),input.var_qual_text_sup_color()]]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="var_qual_text_actif_color",label="Variables qualitatives actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.var_quant_text_actif_color(),input.var_qual_text_sup_color()]]},selected="green")

                @reactive.Effect
                def _():
                    ui.update_select(id="var_qual_text_sup_color",label="Variables qualitatives supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.var_quant_text_actif_color(),input.var_qual_text_actif_color()]]},selected="red")
            else:
                @reactive.Effect
                def _():
                    ui.update_select(id="var_quant_text_actif_color",label="variables quantitatives actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.var_qual_text_actif_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="var_qual_text_actif_color",label="Variables qualitatives actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.var_quant_text_actif_color()]},selected="green")
            
            #-----------------------------------------------------------------------------------------
            ## Individuals - FAMD
            #-----------------------------------------------------------------------------------------
            @reactive.Calc
            def plot_ind():
                if hasattr(model,"ind_sup_"):
                    ind_sup = True
                else:
                    ind_sup = False
                
                if hasattr(model,"quali_sup_"):
                    quali_sup = True
                else:
                    quali_sup = False

                if input.ind_text_color() == "actif/sup":
                    if hasattr(model,"ind_sup_"):
                        color_sup = input.ind_text_sup_color()
                    else:
                        color_sup = None
                    
                    if hasattr(model,"quali_sup_"):
                        color_quali_sup = input.ind_text_quali_sup_color()
                    else:
                        color_quali_sup = None
 
                    fig = fviz_famd_ind(self = model,
                                        axis = [int(input.axis1()),int(input.axis2())],
                                        color = input.ind_text_actif_color(),
                                        color_quali_var = input.ind_text_quali_actif_color(),
                                        ind_sup = ind_sup,
                                        quali_sup = quali_sup,
                                        color_sup = color_sup,
                                        color_quali_sup = color_quali_sup,
                                        text_size = input.ind_text_size(),
                                        lim_contrib =input.ind_lim_contrib(),
                                        lim_cos2 = input.ind_lim_cos2(),
                                        title = input.ind_title(),
                                        repel=input.ind_plot_repel())
                elif input.ind_text_color() in ["cos2","contrib"]:
                    fig = fviz_famd_ind(self = model,
                                        axis = [int(input.axis1()),int(input.axis2())],
                                        color = input.ind_text_color(),
                                        ind_sup = ind_sup,
                                        quali_sup = quali_sup,
                                        text_size = input.ind_text_size(),
                                        lim_contrib = input.ind_lim_contrib(),
                                        lim_cos2 = input.ind_lim_cos2(),
                                        title = input.ind_title(),
                                        repel=input.ind_plot_repel())
                elif input.ind_text_color() == "var_qual":
                    fig = fviz_famd_ind(self=model,
                                         axis=[int(input.axis1()),int(input.axis2())],
                                         text_size = input.ind_text_size(),
                                         lim_contrib =input.ind_lim_contrib(),
                                         lim_cos2 = input.ind_lim_cos2(),
                                         title = input.ind_title(),
                                         habillage = input.ind_text_var_qual_color(),
                                         add_ellipses=input.ind_text_add_ellipse(),
                                         ind_sup=ind_sup,
                                         quali_sup=quali_sup,
                                         repel=input.ind_plot_repel())
                elif  input.ind_text_color() == "var_quant":
                    fig  = fviz_famd_ind(self=model,
                                         axis=[int(input.axis1()),int(input.axis2())],
                                         color = input.ind_text_var_quant_color(),
                                         text_size = input.ind_text_size(),
                                         lim_contrib =input.ind_lim_contrib(),
                                         lim_cos2 = input.ind_lim_cos2(),
                                         title = input.ind_title(),
                                         ind_sup = ind_sup,
                                         quali_sup=quali_sup,
                                         repel=input.ind_plot_repel())
                elif input.ind_text_color() == "kmeans":
                    kmeans = KMeans(n_clusters=input.ind_text_kmeans_nb_clusters(), random_state=np.random.seed(123), n_init="auto").fit(model.ind_["coord"])
                    fig = fviz_famd_ind(self = model,
                                       axis = [int(input.axis1()),int(input.axis2())],
                                       color = kmeans,
                                       ind_sup = ind_sup,
                                       quali_sup = quali_sup,
                                       text_size = input.ind_text_size(),
                                       lim_contrib = input.ind_lim_contrib(),
                                       lim_cos2 = input.ind_lim_cos2(),
                                       title = input.ind_title(),
                                       repel=input.ind_plot_repel())
                return fig+pn.theme_gray()

            # Individuals - FAMD
            @render.plot(alt="Individuals - FAMD")
            def fviz_ind_plot():
                return plot_ind().draw()
            
            # import io
            # @session.download(filename="Individuals-Factor-Map.png")
            # def IndGraphDownloadPng():
            #     with io.BytesIO() as buf:
            #         plt.savefig(RowPlot(), format="png")
            #         yield buf.getvalue()

            #-------------------------------------------------------------------------------------------------
            #   Correlation circle - FAMD
            #-------------------------------------------------------------------------------------------------
            @reactive.Calc
            def plot_quanti_var():
                if hasattr(model,"quanti_sup_"):
                    quanti_sup = True
                else:
                    quanti_sup = False
                
                if input.quanti_var_text_color() == "actif/sup":
                    if hasattr(model,"quanti_sup_"):
                        color_sup = input.quanti_var_text_sup_color()
                    else:
                        color_sup = None
                    
                    fig = fviz_famd_col(self = model,
                                        axis = [int(input.axis1()),int(input.axis2())],
                                        title = input.quanti_var_title(),
                                        color = input.quanti_var_text_actif_color(),
                                        quanti_sup = quanti_sup,
                                        color_sup = color_sup,
                                        text_size = input.quanti_var_text_size(),
                                        lim_contrib = input.quanti_var_lim_contrib(),
                                        lim_cos2 = input.quanti_var_lim_cos2())
                elif input.quanti_var_text_color() in ["cos2","contrib"]:
                    fig = fviz_famd_col(self = model,
                                        axis = [int(input.axis1()),int(input.axis2())],
                                        title = input.quanti_var_title(),
                                        color = input.quanti_var_text_color(),
                                        quanti_sup = quanti_sup,
                                        text_size = input.quanti_var_text_size(),
                                        lim_contrib = input.quanti_var_lim_contrib(),
                                        lim_cos2 = input.quanti_var_lim_cos2())
                elif input.quanti_var_text_color() == "kmeans":
                    kmeans = KMeans(n_clusters=input.quanti_var_text_kmeans_nb_clusters(), random_state=np.random.seed(123), n_init="auto").fit(model.quanti_var_["coord"])
                    fig = fviz_famd_col(self = model,
                                       axis = [int(input.axis1()),int(input.axis2())],
                                       title = input.quanti_var_title(),
                                       color = kmeans,
                                       quanti_sup = quanti_sup,
                                       text_size = input.quanti_var_text_size(),
                                       lim_contrib = input.quanti_var_lim_contrib(),
                                       lim_cos2 = input.quanti_var_lim_cos2())
                return fig + pn.theme_gray()
            
            @render.plot(alt="Correlation circle - FAMD")
            def fviz_quanti_var_plot():
                return plot_quanti_var().draw()

            #------------------------------------------------------------------------------------
            #  Variables categories - FAMD
            #-----------------------------------------------------------------------------------
            @reactive.Calc
            def plot_quali_var():
                if hasattr(model,"quali_sup_"):
                    quali_sup = True
                else:
                    quali_sup = False
                if input.quali_var_text_color() == "actif/sup":
                    if hasattr(model,"quali_sup_"):
                        color_sup = input.quali_var_text_sup_color()
                    else:
                        color_sup = None
                    fig = fviz_famd_mod(self = model,
                                        axis = [int(input.axis1()),int(input.axis2())],
                                        title = input.quali_var_title(),
                                        color = input.quali_var_text_actif_color(),
                                        quali_sup = quali_sup,
                                        color_sup = color_sup,
                                        text_size = input.quali_var_text_size(),
                                        lim_contrib = input.quali_var_lim_contrib(),
                                        lim_cos2 = input.quali_var_lim_cos2(),
                                        repel = input.quali_var_plot_repel())
                elif input.quali_var_text_color() in ["cos2","contrib"]:
                    fig = fviz_famd_mod(self = model,
                                        axis = [int(input.axis1()),int(input.axis2())],
                                        title = input.quali_var_title(),
                                        color = input.quali_var_text_color(),
                                        quali_sup = quali_sup,
                                        text_size = input.quali_var_text_size(),
                                        lim_contrib = input.quali_var_lim_contrib(),
                                        lim_cos2 = input.quali_var_lim_cos2(),
                                        repel = input.quali_var_plot_repel())
                elif input.quali_var_text_color() == "kmeans":
                    kmeans = KMeans(n_clusters = input.quali_var_text_kmeans_nb_clusters(), random_state = np.random.seed(123), n_init="auto").fit(model.quali_var_["coord"])
                    fig = fviz_famd_mod(self = model,
                                       axis = [int(input.axis1()),int(input.axis2())],
                                       title = input.quali_var_title(),
                                       color = kmeans,
                                       quali_sup = quali_sup,
                                       text_size = input.quali_var_text_size(),
                                       lim_contrib = input.quali_var_lim_contrib(),
                                       lim_cos2 = input.quali_var_lim_cos2(), 
                                       repel = input.quali_var_plot_repel())
                return fig + pn.theme_gray()
                
            # Variables categories - FAMD
            @render.plot(alt="Variables categories - FAMD")
            def fviz_quali_var_plot():
                return plot_quali_var().draw()
            
            #------------------------------------------------------------------------------------------------
            # Variables Map
            #-------------------------------------------------------------------------------------------------
            @reactive.Calc
            def plot_var():
                if hasattr(model,"quanti_sup_"):
                    color_quanti_sup = input.var_quant_text_sup_color()
                    quanti_sup = True
                else:
                    quanti_sup = False
                    color_quanti_sup = None
                
                if hasattr(model,"quali_sup_"):
                    quali_sup = True
                    color_quali_sup = input.var_qual_text_sup_color()
                else:
                    quali_sup = False
                    color_quali_sup = None
                
                fig = fviz_famd_var(self=model,
                                    axis=[int(input.axis1()),int(input.axis2())],
                                    title = input.var_title(),
                                    color_quali=input.var_qual_text_actif_color(),
                                    color_quanti=input.var_quant_text_actif_color(),
                                    add_quanti_sup=quanti_sup ,
                                    color_quanti_sup=color_quanti_sup,
                                    add_quali_sup=quali_sup,
                                    color_quali_sup=color_quali_sup,
                                    text_size=input.var_text_size(),
                                    repel = input.var_plot_repel(),
                                    ggtheme=pn.theme_gray())
                return fig

            # Variables Factor Map - MCA
            @output
            @render.plot(alt="Variables - FAMD")
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
            
            #-----------------------------------------------------------------------------------------
            ## Quantitative variables informations
            #-----------------------------------------------------------------------------------------
            # Factor coordinates
            @render.data_frame
            def quanti_var_coord_table():
                quanti_var_coord = model.quanti_var_["coord"].round(4).reset_index()
                quanti_var_coord.columns = ["Variables",*quanti_var_coord.columns[1:]]
                return  DataTable(data = match_datalength(quanti_var_coord,input.quanti_var_coord_len()),filters=input.quanti_var_coord_filter())
            
            # Continuous variables contributions
            @render.data_frame
            def quanti_var_contrib_table():
                quanti_var_contrib = model.quanti_var_["contrib"].round(4).reset_index()
                quanti_var_contrib.columns = ["Variables",*quanti_var_contrib.columns[1:]]
                return  DataTable(data=match_datalength(quanti_var_contrib,input.quanti_var_contrib_len()),filters=input.quanti_var_contrib_filter())
            
            # Variables Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.quanti_var_contrib_graph_btn)
            def _():
                graph_modal_show(text="quanti_var",name="contrib",max_axis=model.call_["n_components"])

            # Plot Individuals Contributions
            @reactive.Calc
            def plot_quanti_var_contrib():
                return fviz_contrib(self=model,choice="quanti_var",axis=input.quanti_var_contrib_axis(),top_contrib=int(input.quanti_var_contrib_top()),color = input.quanti_var_contrib_color(),bar_width= input.quanti_var_contrib_bar_width(),ggtheme=pn.theme_gray())

            @render.plot(alt="Quantitative variables contributions Map - FAMD")
            def fviz_quanti_var_contrib():
                return plot_quanti_var_contrib().draw()
            
            # Square cosinus
            @render.data_frame
            def quanti_var_cos2_table():
                quanti_var_cos2 = model.quanti_var_["cos2"].round(4).reset_index()
                quanti_var_cos2.columns = ["Variables",*quanti_var_cos2.columns[1:]]
                return  DataTable(data = match_datalength(quanti_var_cos2,input.quanti_var_cos2_len()),filters=input.quanti_var_cos2_filter())
            
            # Variables Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.quanti_var_cos2_graph_btn)
            def _():
                graph_modal_show(text="quanti_var",name="cos2",max_axis=model.call_["n_components"])
            
            # Plot Individuals Contributions
            @reactive.Calc
            def plot_quanti_var_cos2():
                return fviz_cos2(self=model,choice="quanti_var",axis=input.quanti_var_cos2_axis(),top_cos2=int(input.quanti_var_cos2_top()),color = input.quanti_var_cos2_color(),bar_width= input.quanti_var_cos2_bar_width(),ggtheme=pn.theme_gray())
            
            @render.plot(alt="Quantitative variables cosinus Map - FAMD")
            def fviz_quanti_var_cos2():
                return plot_quanti_var_cos2().draw()
            
            #-----------------------------------------------------------------------------------------
            ## Supplementary quantitative variables
            #-----------------------------------------------------------------------------------------
            if hasattr(model,"quanti_sup_"):
                @render.ui
                def quanti_sup_panel():
                    return ui.panel_conditional("input.value_choice == 'quanti_sup_res'",
                                ui.input_radio_buttons(id="quanti_sup_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                                ui.panel_conditional("input.quanti_sup_choice === 'coord'",panel_conditional1(text="quanti_sup",name="coord")),
                                ui.panel_conditional("input.quanti_sup_choice === 'cos2'",panel_conditional1(text="quanti_sup",name="cos2"))
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
            
            #----------------------------------------------------------------------------------------------------
            ##   Categories/modalités
            #----------------------------------------------------------------------------------------------------
            # Fcator coordinates
            @render.data_frame
            def quali_var_coord_table():
                quali_var_coord = model.quali_var_["coord"].round(4).reset_index()
                quali_var_coord.columns = ["Categories",*quali_var_coord.columns[1:]]
                return DataTable(data=match_datalength(data=quali_var_coord,value=input.quali_var_coord_len()),filters=input.quali_var_coord_filter())
            
            # Variables Contributions
            @render.data_frame
            def quali_var_contrib_table():
                quali_var_contrib = model.quali_var_["contrib"].round(4).reset_index()
                quali_var_contrib.columns = ["Categories",*quali_var_contrib.columns[1:]]
                return  DataTable(data=match_datalength(data=quali_var_contrib,value=input.quali_var_contrib_len()),filters=input.quali_var_contrib_filter())
            
            # Add Variables Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.quali_var_contrib_graph_btn)
            def _():
                graph_modal_show(text="quali_var",name="contrib",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def plot_quali_var_contrib():
                return fviz_contrib(self=model,choice="quali_var",axis=input.quali_var_contrib_axis(),top_contrib=int(input.quali_var_contrib_top()),color=input.quali_var_contrib_color(),bar_width=input.quali_var_contrib_bar_width(),ggtheme=pn.theme_gray())

            # Plot variables Contributions
            @render.plot(alt="Variables/categories contributions Map - FAMD")
            def fviz_quali_var_contrib():
                return plot_quali_var_contrib().draw()
            
            # Square cosinus
            @render.data_frame
            def quali_var_cos2_table():
                quali_var_cos2 = model.quali_var_["cos2"].round(4).reset_index()
                quali_var_cos2.columns = ["Categories",*quali_var_cos2.columns[1:]]
                return  DataTable(data=match_datalength(data=quali_var_cos2,value=input.quali_var_cos2_len()),filters=input.quali_var_cos2_filter())
            
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.quali_var_cos2_graph_btn)
            def _():
                graph_modal_show(text="quali_var",name="cos2",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def plot_quali_var_cos2():
                return fviz_cos2(self=model,choice="quali_var",axis=input.quali_var_cos2_axis(),top_cos2=int(input.quali_var_cos2_top()),color=input.quali_var_cos2_color(),bar_width=input.quali_var_cos2_bar_width(),ggtheme=pn.theme_gray())

            # Plot variables categories Cos2
            @render.plot(alt="Variables/categories Cosines Map - FAMD")
            def fviz_quali_var_cos2():
                return plot_quali_var_cos2().draw()
            
            # Value - test
            @render.data_frame
            def quali_var_vtest_table():
                quali_var_vtest = model.quali_var_["vtest"].round(4).reset_index()
                quali_var_vtest.columns = ["Categories",*quali_var_vtest.columns[1:]]
                return  DataTable(data=match_datalength(data=quali_var_vtest,value=input.quali_var_vtest_len()),filters=input.quali_var_vtest_filter())
            
            #------------------------------------------------------------------------------------------
            # Supplementary qualitatives variables
            #------------------------------------------------------------------------------------------
            if hasattr(model,"quali_sup_"):
                @render.ui
                def quali_sup_panel():
                    return ui.panel_conditional("input.value_choice == 'quali_sup_res'",
                                ui.input_radio_buttons(id="quali_sup_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","cos2":"Cos2 - Qualité de la représentation","vtest":"Value-test","eta2" : "Eta2 - Rapport de corrélation"},selected="coord",width="100%",inline=True),
                                ui.panel_conditional("input.quali_sup_choice === 'coord'",panel_conditional1(text="quali_sup",name="coord")),
                                ui.panel_conditional("input.quali_sup_choice === 'cos2'",panel_conditional1(text="quali_sup",name="cos2")),
                                ui.panel_conditional("input.quali_sup_choice === 'vtest'",panel_conditional1(text="quali_sup",name="vtest")),
                                ui.panel_conditional("input.quali_sup_choice === 'eta2'",panel_conditional1(text="quali_sup",name="eta2"))
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
            
            #-------------------------------------------------------------------------------------------
            ##   Variables informations
            #------------------------------------------------------------------------------------------
            # Factor coordinates
            @render.data_frame
            def var_coord_table():
                var_coord = model.var_["coord"].round(4).reset_index()
                var_coord.columns = ["Variables",*var_coord.columns[1:]]
                return DataTable(data = match_datalength(var_coord,input.var_coord_len()),filters=input.var_coord_filter())
            
            # Contributions
            @render.data_frame
            def var_contrib_table():
                var_contrib = model.var_["contrib"].round(4).reset_index()
                var_contrib.columns = ["Variables",*var_contrib.columns[1:]]
                return  DataTable(data=match_datalength(var_contrib,input.var_contrib_len()),filters=input.var_contrib_filter())
            
            # Square cosinus
            @render.data_frame
            def var_cos2_table():
                var_cos2 = model.var_["cos2"].round(4).reset_index()
                var_cos2.columns = ["Variables",*var_cos2.columns[1:]]
                return  DataTable(data=match_datalength(var_cos2,input.var_cos2_len()),filters=input.var_cos2_filter())

            #--------------------------------------------------------------------------------------------------------
            ## Individuals informations
            #--------------------------------------------------------------------------------------------------------
            # Factor coordinates
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
                return fviz_contrib(self=model,choice="ind",axis=input.ind_contrib_axis(),top_contrib=int(input.ind_contrib_top()),color = input.ind_contrib_color(),bar_width= input.ind_contrib_bar_width(),ggtheme=pn.theme_gray())

            # Plot Individuals Contributions
            @render.plot(alt="Individuals Contributions Map - FAMD")
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
                graph_modal_show(text="ind",name="cos2",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def ind_cos2_plot():
                return fviz_cos2(self=model,choice="ind",axis=input.ind_cos2_axis(),top_cos2=int(input.ind_cos2_top()),color=input.ind_cos2_color(),bar_width=input.ind_cos2_bar_width(),ggtheme=pn.theme_gray())

            # Plot variables Cos2
            @render.plot(alt="Individuals Cosines Map - FAMD")
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
                                ui.panel_conditional("input.ind_sup_choice === 'coord'",panel_conditional1(text="ind_sup",name="coord")),
                                ui.panel_conditional("input.ind_sup_choice === 'cos2'",panel_conditional1(text="ind_sup",name="cos2"))
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
            
            #----------------------------------------------------------------------------------------
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
                            ui.panel_conditional("input.dim_desc_choice === 'quanti'",panel_conditional1(text="quanti",name="desc")),
                            ui.panel_conditional("input.dim_desc_choice === 'quali'",panel_conditional1(text="quali",name="desc"))
                        )
                    elif "quanti" in Dimdesc.keys() and "quali" not in Dimdesc.keys():
                        return ui.TagList(
                            ui.input_radio_buttons(id="dim_desc_choice",label=ui.h6("Choice"),choices={"quanti" : "Quantitative"},selected="quanti",width="100%",inline=True),
                            ui.panel_conditional("input.dim_desc_choice === 'quanti'",panel_conditional1(text="quanti",name="desc"))
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
            # Quantitative data
            @reactive.Calc
            def quanti_data():
                data = model.call_["Xtot"].loc[:,quanti_var_labels].astype("float")
                if hasattr(model,"ind_sup_"):
                    data = data.drop(index=model.call_["ind_sup"])
                return data
            
            # Qualitative data
            @reactive.Calc
            def quali_data():
                data = model.call_["Xtot"].loc[:,quali_var_labels].astype("object")
                if hasattr(model,"ind_sup_"):
                    data = data.drop(index=model.call_["ind_sup"])
                return data

            # Descriptive statistics
            @render.data_frame
            def stats_desc_table():
                data = model.call_["Xtot"]
                if hasattr(model,"ind_sup_"):
                    data = data.drop(index=model.call_["ind_sup"])
                stats_desc = data.describe(include="all").round(4).T.reset_index().rename(columns={"index":"Variables"})
                return  DataTable(data = match_datalength(stats_desc,input.stats_desc_len()),filters=input.stats_desc_filter())
            
            # Histogram plot
            @reactive.Calc
            def plot_hist():
                p = pn.ggplot(quanti_data(),pn.aes(x=input.quanti_var_label()))
                # Add density
                if input.add_density():
                    p = p + pn.geom_histogram(pn.aes(y="..density.."), color="black", fill="gray")+pn.geom_density(alpha=.2, fill="#FF6666")
                else:
                    p = p + pn.geom_histogram(color="black", fill="gray")
                return p + pn.ggtitle(f"Histogram de {input.quanti_var_label()}")

            @render.plot(alt="Histogram - FAMD")
            def fviz_hist_plot():
                return plot_hist().draw()

            # Pearson correlation matrix
            @render.data_frame
            def corr_matrix_table():
                corr_mat = quanti_data().corr(method="pearson").round(4).reset_index().rename(columns={"index":"Variables"})
                return DataTable(data = match_datalength(corr_mat,input.corr_matrix_len()),filters=input.corr_matrix_filter())

            # Bar plot
            @reactive.Calc
            def plot_bar():
                return pn.ggplot(quali_data(),pn.aes(x=input.quali_var_label()))+pn.geom_bar(color="black", fill="gray")
            
            @render.plot(alt="Bar-Plot")
            def fviz_bar_plot():
                return plot_bar().draw()
            
            # Chi2 test
            @render.data_frame
            def chi2_test_table():
                chi2_test = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"])
                idx = 0
                for i in np.arange(quali_data().shape[1]-1):
                    for j in np.arange(i+1,quali_data().shape[1]):
                        tab = pd.crosstab(quali_data().iloc[:,i],quali_data().iloc[:,j])
                        statistic, pvalue,dof,_ = sp.stats.chi2_contingency(observed=tab,correction=False)
                        row_others = pd.DataFrame({"variable1" : quali_data().columns[i],
                                                   "variable2" : quali_data().columns[j],
                                                   "statistic"    : round(statistic,4),
                                                   "dof" : int(dof),
                                                   "pvalue"   : round(pvalue,4)},
                                                   index=[idx])
                        chi2_test = pd.concat((chi2_test,row_others),axis=0,ignore_index=True)
                        idx = idx + 1
                return  DataTable(data = match_datalength(chi2_test,input.chi2_test_len()),filters=input.chi2_test_filter())
            
            # Others tests
            @render.data_frame
            def others_test_table():
                others_test = pd.DataFrame(columns=["variable1","variable2","cramer","tschuprow","pearson"])
                idx = 0
                for i in np.arange(quali_data().shape[1]-1):
                    for j in np.arange(i+1,quali_data().shape[1]):
                        tab = pd.crosstab(quali_data().iloc[:,i],quali_data().iloc[:,j])
                        row_others = pd.DataFrame({"variable1" : quali_data().columns[i],
                                                   "variable2" : quali_data().columns[j],
                                                   "cramer"    : round(sp.stats.contingency.association(tab,method="cramer"),4),
                                                   "tschuprow" : round(sp.stats.contingency.association(tab,method="tschuprow"),4),
                                                   "pearson"   : round(sp.stats.contingency.association(tab,method="pearson"),4)},
                                                   index=[idx])
                        others_test = pd.concat((others_test,row_others),axis=0,ignore_index=True)
                        idx = idx + 1
                return  DataTable(data = match_datalength(others_test,input.others_test_len()),filters=input.others_test_filter())
            
            #-------------------------------------------------------------------------------------------------
            # Data
            #-------------------------------------------------------------------------------------------------
            # Overall data
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