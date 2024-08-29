# -*- coding: utf-8 -*-
from shiny import Inputs, Outputs, Session, render, ui, reactive
import shinyswatch
import numpy as np
import scipy as sp
import pandas as pd
import plotnine as pn
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from scientisttools import MCA,fviz_mca_ind,fviz_mca_mod,fviz_mca_var,fviz_eig, fviz_contrib,fviz_cos2,fviz_corrcircle, dimdesc
from scientistshiny.base import Base
from scientistshiny.function import *

class MCAshiny(Base):
    """
    Multiple Correspondence Analysis/Specific Multiple Correspondence Analyis (MCA/SpecificMCA) with scientistshiny
    ---------------------------------------------------------------------------------------------------------------

    Description
    -----------
    Performs Multiple Correspondence Analysis (MCA) or Specific Multiple Correspondence Analysis (SpecificMCA) with supplementary elements (individuals and/or quantitative variables and/or qualitative variables) on a Shiny for Python application. Allows to change MCA/SpecificMCA graphical parameters. Graphics can be downloaded in png, jpg and pdf.

    Usage
    -----
    ```python
    >>> MCAshiny(model)
    ```

    Parameters
    ----------
    `model` : a pandas dataframe with n rows (individuals) and p columns (variables) or an object of class MCA/SpecificMCA (a MCA/SpecificMCA result from scientisttools).

    Returns
    -------
    `Graphs` : a tab containing the individuals factor map and the variables categories factor map

    `Values` : a tab containing the eigenvalue, the results for the variables, the results for the individuals, the results for the supplementary elements (individuals and/or qualitative variables and/or quantitative variables).

    `Automatic description of axes` : a tab containing the output of the dimdesc function. This function is designed to  point out the variables and the categories that are the most characteristic according to each dimension obtained by a Factor Analysis.

    `Summary of dataset` : a tab containing the summary of the dataset and a bar plot, chi2 test for qualitative variables, others association test (cramer's V, tschuprow's T and pearson)

    `Data` : a tab containing the dataset with a nice display.

    The left part of the application allows to change some elements of the graphs (axes, variables, colors,.)

    Author
    ------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load dataset and functions
    >>> from scientisttools import MCA, load_poison
    >>> from scientistshiny import MCAshiny
    >>> poison = load_poison()
    >>> 
    >>> # MCA with scientistshiny
    >>> res_shiny = MCAshiny(model=poison)
    >>> res_shiny.run()
    >>> 
    >>> # MCAshiny on a result of a MCA
    >>> res_mca = MCA(n_components=5,ind_sup=list(range(50,55)),quali_sup = [2,3],quanti_sup =[0,1]).fit(poison)
    >>> res_shiny = MCAshiny(model=res_mca)
    >>> res_shiny.run()
    >>> 
    >>> # MCAshiny on a result of a SpecificMCA
    >>> excl = {"Sick" : "Sick_n", "Sex" : "F"}
    >>> res_smca = SpecificMCA(n_components=5,excl=excl,quanti_sup=[0,1],quali_sup=[13,14]).fit(poison)
    >>> res_shiny = MCAshiny(model=res_smca)
    >>> res_shiny.run()
    ```
    
    for jupyter notebooks
    https://stackoverflow.com/questions/74070505/how-to-run-fastapi-application-inside-jupyter
    """
    def __init__(self,model=None):
        # Check if model is an instance of pd.DataFrame class
        if isinstance(model,pd.DataFrame):        
            # Check if quantitative data
            is_quanti = model.select_dtypes(include=np.number)
            if is_quanti.shape[1]>0:
                for col in is_quanti.columns.tolist():
                    model[col] = model[col].astype("float")
                quanti_sup = [model.columns.tolist().index(x) for x in model.columns if x in is_quanti.columns]
            else:
                quanti_sup = None
            
            # Fit the MCA with scientisttools
            model = MCA(quanti_sup=quanti_sup).fit(model)

        # Check if model is MCA/SpecificMCA
        if model.model_ not in ["mca","specificmca"]:
            raise TypeError("'model' must be an instance of class MCA, SpecificMCA")

        if model.model_ == "mca":
            title = "Analyse des Correspondances Multiples"
            model_name = "MCA"
        else:
            title = "Analyse des Correspondances Multiples Spécifiques"
            model_name = "SpecificMCA"
        
        ind_text_color_choices = {"actif/sup":"actifs/supplémentaires","cos2":"Cosinus","contrib":"Contribution","var_qual":"Variable qualitative","kmeans" : "KMeans"}

        # Plot Choice
        graph_choices = {"fviz_ind":"individus","fviz_mod":"modalités","fviz_var":"variables"}

        # Resume choice
        resume_choices = {"stats_desc":"Statistiques descriptives","bar_plot":"Diagramme en barres","chi2_test" : "Test de Chi2","others_test":"Autres mesures d'association"}

        # Qualitative variables labels
        var_labels = model.call_["X"].columns.tolist()
        
        # Initialise value choice
        value_choice = {"eigen_res":"Valeurs propres","mod_res":"Résultats des modalités","ind_res":"Résultats sur les individus","var_res":"Résultats sur les variables"}
        
        # Check if supplementary individuals
        if hasattr(model,"ind_sup_"):
            value_choice = {**value_choice,**{"ind_sup_res" : "Résultats des individus supplémentaires"}}
        
        # Check if supplementary qualitatives variables
        if hasattr(model,"quali_sup_"):
            value_choice = {**value_choice,**{"quali_sup_res" : "Résultats des variables qualitatives supplémentaires"}}
            var_labels = [*var_labels,*model.quali_sup_["eta2"].index.tolist()]

        # Check if supplementary quantitatives variables
        if hasattr(model,"quanti_sup_"):
            value_choice = {**value_choice, **{"quanti_sup_res" : "Résultats des variables quantitatives supplémentaires"}}
            ind_text_color_choices = {**ind_text_color_choices,**{"var_quant" : "Variable quantitative"}}
            resume_choices = {**resume_choices,**{"hist_plot" : "Histogramme"}}
            graph_choices = {**graph_choices,**{"fviz_quanti_sup" : "Variables quantitatives supplémentaires"}}
            
        # UI
        app_ui = ui.page_fluid(
            ui.include_css(css_path),
            shinyswatch.theme.superhero(),
            header(title=title,model_name=model_name),
            ui.page_sidebar(
                ui.sidebar(
                    ui.panel_well(
                        ui.h6("Options graphiques",style="text-align:center"),
                        ui.div(ui.h6("Axes"),style="display: inline-block;padding: 5px"),
                        axes_input_select(model=model),
                        ui.br(),
                        ui.div(ui.input_select(id="fviz_choice",label="Modifier le graphe des",choices=graph_choices,selected="fviz_ind",multiple=False,width="100%")),
                        ui.panel_conditional("input.fviz_choice === 'fviz_ind'",
                            title_input(id="ind_title",value="Individuals - MCA"),
                            text_size_input(which="ind"),
                            point_select_input(id="ind_point_select"),
                            ui.panel_conditional("input.ind_point_select === 'cos2'",ui.div(lim_cos2("ind_lim_cos2"),align="center")),
                            ui.panel_conditional("input.ind_point_select === 'contrib'",ui.div(lim_contrib("ind_lim_contrib"),align="center")              ),
                            text_color_input(id="ind_text_color",choices=ind_text_color_choices),
                            ui.panel_conditional("input.ind_text_color === 'actif/sup'",
                                ui.input_select(id="ind_text_actif_color",label="Individus actifs",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                                ui.output_ui("ind_text_sup")
                            ),
                            ui.panel_conditional("input.ind_text_color === 'kmeans'",ui.input_numeric(id="ind_text_kmeans_nb_clusters",label="Choix du nombre de clusters",value=2,min=1,max=model.ind_["coord"].shape[0],step=1,width="100%")),
                            ui.panel_conditional("input.ind_text_color === 'var_qual'",
                                ui.input_select(id="ind_text_var_qual_color",label="choix de la variable",choices={x:x for x in var_labels},selected=var_labels[0],multiple=False,width="100%"),
                                ui.input_switch(id="ind_text_add_ellipse",label="Trace les ellipses de confiance autour des barycentres",value=False)
                            ),
                            ui.panel_conditional("input.ind_text_color === 'var_quant'",ui.output_ui("ind_text_var_quant")),
                            ui.input_switch(id="ind_plot_repel",label="repel",value=True)
                        ),
                        ui.panel_conditional("input.fviz_choice === 'fviz_mod'",
                            title_input(id="mod_title",value="Variables categories - MCA"),
                            text_size_input(which="mod"),
                            point_select_input(id="mod_point_select"),
                            ui.panel_conditional("input.mod_point_select === 'cos2'",ui.div(lim_cos2(id="mod_lim_cos2"),align="center")),
                            ui.panel_conditional("input.mod_point_select === 'contrib'",ui.div(lim_contrib(id="mod_lim_contrib"),align="center")),
                            text_color_input(id="mod_text_color",choices={"actif/sup":"actifs/supplémentaires","cos2":"Cosinus","contrib":"Contribution","kmeans":"KMeans"}),
                            ui.panel_conditional("input.mod_text_color === 'actif/sup'",
                                ui.input_select(id="mod_text_actif_color",label="Modalités actives",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                                ui.output_ui("mod_text_sup")
                            ),
                            ui.panel_conditional("input.mod_text_color === 'kmeans'",ui.input_numeric(id="mod_text_kmeans_nb_clusters",label="Choix du nombre de clusters",value=2,min=1,max=model.var_["coord"].shape[0],step=1,width="100%")),
                            ui.input_switch(id="mod_plot_repel",label="repel",value=True)
                        ),
                        ui.panel_conditional("input.fviz_choice === 'fviz_var'",
                            title_input(id="var_title",value="Variables - MCA"),
                            text_size_input(which="var"),
                            point_select_input(id="var_point_select"),
                            ui.panel_conditional("input.var_point_select === 'cos2'",ui.div(lim_cos2(id="var_lim_cos2"),align="center")),
                            ui.panel_conditional("input.var_point_select === 'contrib'",ui.div(lim_contrib(id="var_lim_contrib"),align="center")),
                            ui.input_select(id="var_text_actif_color",label="Variables actives",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                            ui.output_ui("var_text_quali_sup"),
                            ui.output_ui("var_text_quanti_sup"),
                            ui.input_switch(id="var_plot_repel",label="repel",value=True)
                        ),
                        ui.output_ui("quanti_sup_fviz")
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
                                ui.div(ui.output_plot("fviz_mod_plot",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                                ui.div(ui.download_button(id="download_mod_plot_jpg",label="jpg",style = download_btn_style,icon=None),style="display: inline-block;",align="center"),
                                ui.div(ui.download_button(id="download_mod_plot_png",label="png",style = download_btn_style),style="display: inline-block;",align="center"),
                                ui.div(ui.download_button(id="download_mod_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;",align="center"),
                                align="center"
                            )
                        ),
                        ui.br(),
                        ui.row(
                            ui.column(6,
                                ui.div(ui.output_plot("fviz_var_plot",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px"),
                                ui.div(ui.download_button(id="download_var_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_var_plot_png",label="png",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_var_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                align="center"
                            ),
                            ui.output_ui("quanti_sup_plot")
                        ),
                    ),
                    ui.nav_panel("Valeurs",
                        ui.input_radio_buttons(id="value_choice",label=ui.h6("Quelles sorties voulez-vous?"),choices=value_choice,inline=True),
                        ui.br(),
                        eigen_panel(),
                        ui.panel_conditional("input.value_choice === 'mod_res'",
                            ui.input_radio_buttons(id="mod_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","contrib":"Contributions","cos2":"Cos2 - Qualité de la représentation","vtest":"Value - test"},selected="coord",width="100%",inline=True),
                            ui.panel_conditional("input.mod_choice === 'coord'",panel_conditional1(text="mod",name="coord")),
                            ui.panel_conditional("input.mod_choice === 'contrib'",panel_conditional2(text="mod",name="contrib")),
                            ui.panel_conditional("input.mod_choice === 'cos2'",panel_conditional2(text="mod",name="cos2")),
                            ui.panel_conditional("input.mod_choice === 'vtest'",panel_conditional1(text="mod",name="vtest"))
                        ),
                        ui.panel_conditional("input.value_choice === 'ind_res'",
                            ui.input_radio_buttons(id="ind_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","contrib":"Contributions","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                            ui.panel_conditional("input.ind_choice === 'coord'",panel_conditional1(text="ind",name="coord")),
                            ui.panel_conditional("input.ind_choice === 'contrib'",panel_conditional2(text="ind",name="contrib")),
                            ui.panel_conditional("input.ind_choice === 'cos2'",panel_conditional2(text="ind",name="cos2"))
                        ),
                        ui.panel_conditional(f"input.value_choice == 'var_res'",
                            ui.input_radio_buttons(id="var_choice",label=ui.h6("Quel type de résultats?"),choices={"eta2":"Eta2 - Rapport de corrélation","contrib":"Contributions"},selected="eta2",width="100%",inline=True),
                            ui.panel_conditional("input.var_choice === 'eta2'",panel_conditional1(text="var",name="eta2")),
                            ui.panel_conditional("input.var_choice === 'contrib'",panel_conditional1(text="var",name="contrib"))
                        ),
                        ui.output_ui("ind_sup_panel"),
                        ui.output_ui("quali_sup_panel"),  
                        ui.output_ui("quanti_sup_panel")
                    ),
                    dim_desc_panel(model=model),
                    ui.nav_panel("Résumé du jeu de données",
                        ui.input_radio_buttons(id="resume_choice",label=ui.h6("Quelles sorties voulez - vous?"),choices=resume_choices,selected="stats_desc",width="100%",inline=True),
                        ui.br(),
                        ui.panel_conditional("input.resume_choice === 'stats_desc'",panel_conditional1(text="stats",name="desc")),
                        ui.panel_conditional("input.resume_choice === 'bar_plot'",
                            ui.row(
                                ui.column(2,ui.input_select(id="var_qual_label",label=ui.h6("Choisir une variable"),choices={x:x for x in var_labels},selected=var_labels[0])),
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
                        ),
                        ui.panel_conditional("input.resume_choice === 'chi2_test'",panel_conditional1(text="chi2",name="test")),
                        ui.panel_conditional("input.resume_choice === 'others_test'",panel_conditional1(text="others",name="test")),
                        ui.output_ui("quanti_sup_graph")
                    ),
                    ui.nav_panel("Données",panel_conditional1(text="overall",name="data")
                    )
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
                    return ui.TagList(ui.input_select(id="ind_text_sup_color",label="Individus supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="blue",multiple=False))
            
                # Disable colors
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_actif_color",label="Individus actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.ind_text_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_sup_color",label="Individus supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.ind_text_actif_color()]},selected="blue")
            
            #---------------------------------------------------------------------------------------------------------------
            if hasattr(model,"quanti_sup_"):
                @render.ui
                def ind_text_var_quant():
                    quanti_sup_labels = model.quanti_sup_["coord"].index.tolist()
                    return ui.TagList(ui.input_select(id="ind_text_var_quant_color",label="Choix de la variable",choices={x:x for x in quanti_sup_labels},selected=quanti_sup_labels[0],multiple=False))
                
            # #-------------------------------------------------------------------------------------------
            if hasattr(model,"quali_sup_"):
                @render.ui
                def mod_text_sup():
                    return ui.TagList(ui.input_select(id="mod_text_sup_color",label="Modalités supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="blue",multiple=False,width="100%"))
           
                # Disabled Variables Categories Text Colors
                @reactive.Effect
                def _():
                    ui.update_select(id="mod_text_actif_color",label="Modalités actives",choices={x : x for x in [i for i in mcolors.CSS4_COLORS if i != input.mod_text_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="mod_text_sup_color",label="Modalités supplémentaires",choices={x : x for x in [i for i in mcolors.CSS4_COLORS if i != input.mod_text_actif_color()]},selected="blue")
            
            #------------------------------------------------------------------------------------------
            if hasattr(model,"quali_sup_"):
                @render.ui
                def var_text_quali_sup():
                    return ui.TagList(ui.input_select(id="var_text_sup_color",label="Variables qualitatives supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="blue",multiple=False,width="100%"))
            
            if hasattr(model,"quanti_sup_"):
                @render.ui
                def var_text_quanti_sup():
                    return ui.TagList(ui.input_select(id="var_text_quanti_sup_color",label="Variables quantitatives supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="red",multiple=False,width="100%"))
            
            #-------------------------------------------------------------------------------------------------------
            # Disable colors
            if hasattr(model,"quali_sup_") and hasattr(model,"quanti_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="var_text_actif_color",label="Variables actives",choices={x : x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.var_text_sup_color(),input.var_text_quanti_sup_color()]]},selected="black")
            
                @reactive.Effect
                def _():
                    ui.update_select(id="var_text_sup_color",label="Variables qualitatives supplémentaires",choices={x : x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.var_text_actif_color(),input.var_text_quanti_sup_color()]]},selected="blue")

                @reactive.Effect
                def _():
                    ui.update_select(id="var_text_quanti_sup_color",label="Variables quantitatives supplementaires",choices={x : x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.var_text_actif_color(),input.var_text_sup_color()]]},selected="red")
            elif hasattr(model,"quali_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="var_text_actif_color",label="Variables actives",choices={x : x for x in [i for i in mcolors.CSS4_COLORS if i != input.var_text_sup_color()]},selected="black")
            
                @reactive.Effect
                def _():
                    ui.update_select(id="var_text_sup_color",label="Variables qualitatives supplémentaires",choices={x : x for x in [i for i in mcolors.CSS4_COLORS if i != input.var_text_actif_color()]},selected="blue")
            elif hasattr(model,"quanti_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="var_text_actif_color",label="Variables actives",choices={x : x for x in [i for i in mcolors.CSS4_COLORS if i != input.var_text_quanti_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="var_text_quanti_sup_color",label="Variables quantitatives supplementaires",choices={x : x for x in [i for i in mcolors.CSS4_COLORS if i != input.var_text_actif_color()]},selected="red")

            #-----------------------------------------------------------------------------------------
            ## Individuals MCA
            #-----------------------------------------------------------------------------------------
            @reactive.Calc
            def plot_ind():
                if hasattr(model,"ind_sup_"):
                    ind_sup = True
                else:
                    ind_sup = False
                
                if input.ind_text_color() == "actif/sup":
                    if hasattr(model,"ind_sup_"):
                        color_sup = input.ind_text_sup_color()
                    else:
                        color_sup = None
                    
                    fig = fviz_mca_ind(self=model,
                                       axis=[int(input.axis1()),int(input.axis2())],
                                       color = input.ind_text_actif_color(),
                                       ind_sup = ind_sup,
                                       color_sup = color_sup,
                                       text_size = input.ind_text_size(),
                                       lim_contrib = input.ind_lim_contrib(),
                                       lim_cos2 = input.ind_lim_cos2(),
                                       title = input.ind_title(),
                                       repel = input.ind_plot_repel())
                elif input.ind_text_color() in ["cos2","contrib"]:
                    fig = fviz_mca_ind(self=model,
                                       axis=[int(input.axis1()),int(input.axis2())],
                                       color=input.ind_text_color(),
                                       ind_sup=ind_sup,
                                       text_size = input.ind_text_size(),
                                       lim_contrib =input.ind_lim_contrib(),
                                       lim_cos2 = input.ind_lim_cos2(),
                                       title = input.ind_title(),
                                       repel = input.ind_plot_repel())
                elif input.ind_text_color() == "kmeans":
                    kmeans = KMeans(n_clusters=input.ind_text_kmeans_nb_clusters(), random_state=np.random.seed(123), n_init="auto").fit(model.ind_["coord"])
                    fig = fviz_mca_ind(
                            self = model,
                            axis = [int(input.axis1()),int(input.axis2())],
                            color = kmeans,
                            ind_sup = ind_sup,
                            text_size = input.ind_text_size(),
                            lim_contrib = input.ind_lim_contrib(),
                            lim_cos2 = input.ind_lim_cos2(),
                            title = input.ind_title(),
                            repel=input.ind_plot_repel()
                        )
                elif input.ind_text_color() == "var_qual":
                    fig = fviz_mca_ind(self = model,
                                       axis = [int(input.axis1()),int(input.axis2())],
                                       ind_sup = ind_sup,
                                       text_size = input.ind_text_size(),
                                       lim_contrib =input.ind_lim_contrib(),
                                       lim_cos2 = input.ind_lim_cos2(),
                                       title = input.ind_title(),
                                       habillage = input.ind_text_var_qual_color(),
                                       add_ellipses = input.ind_text_add_ellipse(),
                                       repel = input.ind_plot_repel())
                elif  input.ind_text_color() == "var_quant":
                    fig = fviz_mca_ind(self = model,
                                       axis = [int(input.axis1()),int(input.axis2())],
                                       color = input.ind_text_var_quant_color(),
                                       legend_title = input.ind_text_var_quant_color(),
                                       text_size = input.ind_text_size(),
                                       lim_contrib =input.ind_lim_contrib(),
                                       lim_cos2 = input.ind_lim_cos2(),
                                       title = input.ind_title(),
                                       repel = input.ind_plot_repel())
                return fig+pn.theme_gray()

            # Individuals - MCA
            @render.plot(alt="Individuals - MCA")
            def fviz_ind_plot():
                return plot_ind().draw()
            
            # import io
            # @render.download(filename="individuals_factor_map.png")
            # def donwload_ind_plot_png():
            #     with io.BytesIO() as buf:
            #         plt.savefig(plot_ind(), format="png")
            #         yield buf.getvalue()
            
            #------------------------------------------------------------------------------------
            ##  Variables categories - MCA
            #------------------------------------------------------------------------------------
            @reactive.Calc
            def plot_mod():
                if hasattr(model,"quali_sup_"):
                    quali_sup = True
                else:
                    quali_sup = False
                
                if input.mod_text_color() == "actif/sup":
                    if hasattr(model,"quali_sup_"):
                        color_sup = input.mod_text_sup_color()
                    else:
                        color_sup = None
                    
                    fig = fviz_mca_mod(self = model,
                                       axis = [int(input.axis1()),int(input.axis2())],
                                       title = input.mod_title(),
                                       color = input.mod_text_actif_color(),
                                       quali_sup = quali_sup,
                                       color_sup = color_sup,
                                       text_size = input.mod_text_size(),
                                       lim_contrib = input.mod_lim_contrib(),
                                       lim_cos2 = input.mod_lim_cos2(),
                                       repel = input.mod_plot_repel())
                elif input.mod_text_color() in ["cos2","contrib"]:
                    fig = fviz_mca_mod(self = model,
                                       axis = [int(input.axis1()),int(input.axis2())],
                                       title = input.mod_title(),
                                       color = input.mod_text_color(),
                                       quali_sup = quali_sup,
                                       text_size = input.mod_text_size(),
                                       lim_contrib = input.mod_lim_contrib(),
                                       lim_cos2 = input.mod_lim_cos2(),
                                       repel = input.mod_plot_repel())
                elif input.mod_text_color() == "kmeans":
                    kmeans = KMeans(n_clusters = input.mod_text_kmeans_nb_clusters(), random_state = np.random.seed(123), n_init="auto").fit(model.var_["coord"])
                    fig = fviz_mca_mod(self = model,
                                       axis = [int(input.axis1()),int(input.axis2())],
                                       title = input.mod_title(),
                                       color = kmeans,
                                       quali_sup = quali_sup,
                                       text_size = input.mod_text_size(),
                                       lim_contrib = input.mod_lim_contrib(),
                                       lim_cos2 = input.mod_lim_cos2(), 
                                       repel = input.mod_plot_repel())
                return fig+pn.theme_gray()

            # Variables categories - MCA
            @render.plot(alt="Variables categories - MCA")
            def fviz_mod_plot():
                return plot_mod().draw()
            
            #-------------------------------------------------------------------------------------------------
            ## Variables - MCA
            #-------------------------------------------------------------------------------------------------
            @reactive.Calc
            def plot_var():
                if hasattr(model,"quali_sup_"):
                    quali_sup = True
                    color_sup = input.var_text_sup_color()
                else:
                    quali_sup = False
                    color_sup = None
                
                if hasattr(model,"quanti_sup_"):
                    quanti_sup = True
                    color_quanti_sup = input.var_text_quanti_sup_color()
                else:
                    quanti_sup = False
                    color_quanti_sup = None

                fig = fviz_mca_var(self = model,
                                   axis = [int(input.axis1()),int(input.axis2())],
                                   title = input.var_title(),
                                   color = input.var_text_actif_color(),
                                   add_quali_sup = quali_sup,
                                   color_sup = color_sup,
                                   add_quanti_sup = quanti_sup,
                                   color_quanti_sup = color_quanti_sup,
                                   text_size = input.var_text_size(),
                                   repel = input.var_plot_repel())
                return fig + pn.theme_gray()

            @render.plot(alt="Variables - MCA")
            def fviz_var_plot():
                return plot_var().draw()
            
            #-------------------------------------------------------------------------------------------------
            ##   Supplementary quantitative variables
            #-------------------------------------------------------------------------------------------------
            if hasattr(model,"quanti_sup_"):
                @render.ui
                def quanti_sup_fviz():
                    return ui.panel_conditional("input.fviz_choice === 'fviz_quanti_sup'",
                            title_input(id="quanti_sup_title",value="Correlation circle - MCA"),
                            text_size_input(which="quanti_sup"),
                            ui.input_select(id="quanti_sup_color",label="Variables quantitatives supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="red",multiple=False,width="100%")
                        )
            
                @render.ui
                def quanti_sup_plot():
                    return ui.TagList(
                            ui.column(6,
                                ui.div(ui.output_plot("fviz_quanti_sup_plot",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                                ui.div(ui.download_button(id="download_quanti_sup_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;",align="center"),
                                ui.div(ui.download_button(id="download_quanti_sup_plot_png",label="png",style = download_btn_style),style="display: inline-block;",align="center"),
                                ui.div(ui.download_button(id="download_quanti_sup_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;",align="center"),
                                align="center"
                            )
                        )

                @reactive.Calc
                def plot_quanti_sup():
                    fig =  fviz_corrcircle(self=model,axis=[int(input.axis1()),int(input.axis2())],color=input.quanti_sup_color(),title=input.quanti_sup_title(),text_size=input.quanti_sup_text_size(),ggtheme=pn.theme_gray())
                    return fig 
                
                @render.plot(alt="Correlation circle - MCA")
                def fviz_quanti_sup_plot():
                    return plot_quanti_sup().draw()

            #-----------------------------------------------------------------------------------------------------
            ## Eigenvalue informations
            #-----------------------------------------------------------------------------------------------------
            # Reactive
            @reactive.Calc
            def plot_eigen():
                return fviz_eig(self=model,choice=input.fviz_eigen_choice(),add_labels=input.fviz_eigen_label(),ggtheme=pn.theme_gray())

            @render.plot(alt="Scree Plot - MCA")
            def fviz_eigen(): 
                return plot_eigen().draw()
            
            # Eigen value - DataFrame
            @render.data_frame
            def eigen_table():
                eig = model.eig_.round(4).reset_index().rename(columns={"index":"dimensions"})
                eig.columns = [x.capitalize() for x in eig.columns]
                return DataTable(data=match_datalength(eig,input.eigen_table_len()),filters=input.eigen_table_filter())
            
            #---------------------------------------------------------------------------------------------
            ## Variables categories informations
            #---------------------------------------------------------------------------------------------
            # Factor Coordinates
            @render.data_frame
            def mod_coord_table():
                mod_coord = model.var_["coord"].round(4).reset_index()
                mod_coord.columns = ["Categories", *mod_coord.columns[1:]]
                return DataTable(data=match_datalength(data=mod_coord,value=input.mod_coord_len()),filters=input.mod_coord_filter())
            
            # Contributions
            @render.data_frame
            def mod_contrib_table():
                mod_contrib = model.var_["contrib"].round(4).reset_index()
                mod_contrib.columns = ["Categories", *mod_contrib.columns[1:]]
                return  DataTable(data=match_datalength(data=mod_contrib,value=input.mod_contrib_len()),filters=input.mod_contrib_filter())
            
            # Add Variables Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.mod_contrib_graph_btn)
            def _():
                graph_modal_show(text="mod",name="contrib",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def mod_contrib_plot():
                fig = fviz_contrib(self=model,choice="var",axis=input.mod_contrib_axis(),top_contrib=int(input.mod_contrib_top()),color=input.mod_contrib_color(),bar_width=input.mod_contrib_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot variables Contributions
            @render.plot(alt="Variables categories contributions Map - MCA")
            def fviz_mod_contrib():
                return mod_contrib_plot().draw()
            
            # Square cosinus
            @render.data_frame
            def mod_cos2_table():
                mod_cos2 = model.var_["cos2"].round(4).reset_index()
                mod_cos2.columns = ["Categories", *mod_cos2.columns[1:]]
                return  DataTable(data=match_datalength(data=mod_cos2,value=input.mod_cos2_len()),filters=input.mod_cos2_filter())
            
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.mod_cos2_graph_btn)
            def _():
                graph_modal_show(text="mod",name="cos2",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def mod_cos2_plot():
                fig = fviz_cos2(self=model,choice = "var",axis=input.mod_cos2_axis(),top_cos2=int(input.mod_cos2_top()),color=input.mod_cos2_color(),bar_width=input.mod_cos2_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot variables categories Cos2
            @render.plot(alt="Variables categories Cosines Map - MCA")
            def fviz_mod_cos2():
                return mod_cos2_plot().draw()

            # Value - test
            @render.data_frame
            def mod_vtest_table():
                mod_vtest = model.var_["vtest"].round(4).reset_index()
                mod_vtest.columns = ["Categories", *mod_vtest.columns[1:]]
                return  DataTable(data=match_datalength(data=mod_vtest,value=input.mod_vtest_len()),filters=input.mod_vtest_filter())
            
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
            
            #---------------------------------------------------------------------------------
            ## Supplementary quantitative Variables
            #---------------------------------------------------------------------------------
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
            
            #---------------------------------------------------------------------------------------------
            ## Individuals informations
            #---------------------------------------------------------------------------------------------
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

            # Plot Individuals Contributions
            @reactive.Calc
            def ind_contrib_plot():
                fig = fviz_contrib(self=model,choice="ind",axis=input.ind_contrib_axis(),top_contrib=int(input.ind_contrib_top()),color = input.ind_contrib_color(),bar_width= input.ind_contrib_bar_width(),ggtheme=pn.theme_gray())
                return fig
            
            @render.plot(alt="Individuals Contributions Map - MCA")
            def fviz_ind_contrib():
                return ind_contrib_plot().draw()
        
            # Individuals Cos2 
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

            # Plot variables Cos2
            @reactive.Calc
            def ind_cos2_plot():
                fig = fviz_cos2(self=model,choice="ind",axis=input.ind_cos2_axis(),top_cos2=int(input.ind_cos2_top()),color=input.ind_cos2_color(),bar_width=input.ind_cos2_bar_width(),ggtheme=pn.theme_gray())
                return fig
            
            @render.plot(alt="Individuals Cosines Map - MCA")
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
        
            #------------------------------------------------------------------------------------------
            ##   Variables informations
            #------------------------------------------------------------------------------------------
            # Square correlation ratio
            @render.data_frame
            def var_eta2_table():
                var_eta2 = model.var_["eta2"].round(4).reset_index()
                var_eta2.columns = ["Variables", *var_eta2.columns[1:]]
                return DataTable(data = match_datalength(var_eta2,input.var_eta2_len()),filters=input.var_eta2_filter())
            
            # Variables Contributions
            @render.data_frame
            def var_contrib_table():
                var_contrib = model.var_["var_contrib"].round(4).reset_index()
                var_contrib.columns = ["Variables", *var_contrib.columns[1:]]
                return  DataTable(data=match_datalength(var_contrib,input.var_contrib_len()),filters=input.var_contrib_filter())
            
            #------------------------------------------------------------------------------------------------------------
            ## Description of axis
            #------------------------------------------------------------------------------------------------------------
            @reactive.Effect
            def _():
                Dimdesc = dimdesc(self=model,axis=None,proba=float(input.dim_desc_pvalue()))[input.dim_desc_axis()]

                @output
                @render.ui
                def dim_desc():
                    if "quali" in Dimdesc.keys() and "quanti" in Dimdesc.keys():
                        return ui.TagList(
                            ui.input_radio_buttons(id="dim_desc_choice",label=ui.h6("Choice"),choices={"quali":"Qualitative","quanti":"Quantitative"},selected="quali",width="100%",inline=True),
                            ui.panel_conditional("input.dim_desc_choice === 'quali'",panel_conditional1(text="quali",name="desc")),
                            ui.panel_conditional("input.dim_desc_choice === 'quanti'",panel_conditional1(text="quanti",name="desc"))
                        )
                    elif "quali" in Dimdesc.keys() and "quanti" not in Dimdesc.keys():
                        return ui.TagList(
                            ui.input_radio_buttons(id="dim_desc_choice",label=ui.h6("Choice"),choices={"quali":"Qualitative"},selected="quali",width="100%",inline=True),
                            ui.panel_conditional("input.dim_desc_choice === 'quali'",panel_conditional1(text="quali",name="desc"))
                        )
                    elif "quanti" in Dimdesc.keys() and "quali" not in Dimdesc.keys():
                        return ui.TagList(
                            ui.input_radio_buttons(id="dim_desc_choice",label=ui.h6("Choice"),choices={"quanti":"Quantitative"},selected="quanti",width="100%",inline=True),
                            ui.panel_conditional("input.dim_desc_choice === 'quanti'",panel_conditional1(text="quanti",name="desc"))
                        )
                    else:
                        return ui.TagList(ui.p("No significant variable"))
                
                if "quali" in Dimdesc.keys():
                    @render.data_frame
                    def quali_desc_table():
                        data = Dimdesc["quali"].round(4).reset_index().rename(columns={"index":"Variables"})
                        return  DataTable(data = match_datalength(data,input.quali_desc_len()),filters=input.quali_desc_filter())
                    
                if "quanti" in Dimdesc.keys():
                    @render.data_frame
                    def quanti_desc_table():
                        data = Dimdesc["quanti"].round(4).reset_index().rename(columns={"index":"Variables"})
                        return  DataTable(data = match_datalength(data,input.quanti_desc_len()),filters=input.quanti_desc_filter())

            #-----------------------------------------------------------------------------------------------
            ## Summary of data
            #-----------------------------------------------------------------------------------------------
            @reactive.Calc
            def data():
                data = model.call_["Xtot"].loc[:,var_labels]
                if model.ind_sup is not None:
                    data = data.drop(index=model.call_["ind_sup"])
                return data
            
            # Descriptive statistics
            @render.data_frame
            def stats_desc_table():
                stats_desc = data().describe(include="all").round(4).T.reset_index().rename(columns={"index":"Variables"})
                return  DataTable(data = match_datalength(stats_desc,input.stats_desc_len()),filters=input.stats_desc_filter())

            # Diagramme en barres
            @reactive.Calc
            def bar_plot():
                return pn.ggplot(data(),pn.aes(x=input.var_qual_label()))+ pn.geom_bar(color="black",fill="gray")
            
            @render.plot(alt="Bar-Plot")
            def fviz_bar_plot():
                return bar_plot().draw()

            # Chi2 test
            @render.data_frame
            def chi2_test_table():
                chi2_test = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"])
                idx = 0
                for i in np.arange(data().shape[1]-1):
                    for j in np.arange(i+1,data().shape[1]):
                        tab = pd.crosstab(data().iloc[:,i],data().iloc[:,j])
                        statistic, pvalue,dof,_ = sp.stats.chi2_contingency(observed=tab,correction=False)
                        row_others = pd.DataFrame({"variable1" : data().columns[i],
                                                   "variable2" : data().columns[j],
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
                for i in np.arange(data().shape[1]-1):
                    for j in np.arange(i+1,data().shape[1]):
                        tab = pd.crosstab(data().iloc[:,i],data().iloc[:,j])
                        row_others = pd.DataFrame({"variable1" : data().columns[i],
                                                   "variable2" : data().columns[j],
                                                   "cramer"    : round(sp.stats.contingency.association(tab,method="cramer"),4),
                                                   "tschuprow" : round(sp.stats.contingency.association(tab,method="tschuprow"),4),
                                                   "pearson"   : round(sp.stats.contingency.association(tab,method="pearson"),4)},
                                                   index=[idx])
                        others_test = pd.concat((others_test,row_others),axis=0,ignore_index=True)
                        idx = idx + 1
                return  DataTable(data = match_datalength(others_test,input.others_test_len()),filters=input.others_test_filter())
            
            if hasattr(model,"quanti_sup_"):
                quanti_sup_labels = model.quanti_sup_["coord"].index
                @render.ui
                def quanti_sup_graph():
                    return ui.panel_conditional("input.resume_choice === 'hist_plot'",
                            ui.row(
                                ui.column(2,
                                    ui.input_select(id="quanti_sup_label",label=ui.h6("Choisir une variable"),choices={x:x for x in quanti_sup_labels},selected=quanti_sup_labels[0]),
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
                        )
                
                # Reactive histogram
                @reactive.Calc
                def hist_plot():
                    data = model.call_["Xtot"].loc[:,quanti_sup_labels]
                    if model.ind_sup is not None:
                        data = data.drop(index=model.call_["ind_sup"])
                    p = pn.ggplot(data,pn.aes(x=input.quanti_sup_label()))
                    # Add density
                    if input.add_density():
                        p = p + pn.geom_histogram(pn.aes(y="..density.."), color="black", fill="gray")+pn.geom_density(alpha=.2, fill="#FF6666")
                    else:
                        p = p + pn.geom_histogram(color="black", fill="gray")
                    return p + pn.ggtitle(f"Histogram de {input.quanti_sup_label()}")
                
                # Histogramme
                @render.plot(alt="Histogram")
                def fviz_hist_plot():
                    return hist_plot().draw()
    
            #-------------------------------------------------------------------------------------------------
            ## Overall data
            #-------------------------------------------------------------------------------------------------
            @render.data_frame
            def overall_data_table():
                overalldata = model.call_["Xtot"].reset_index().rename(columns={"index":"Individus"})
                return DataTable(data = match_datalength(overalldata,input.overall_data_len()),filters=input.overall_data_filter())
            
            #-----------------------------------------------------------------------------------------------------------------------
            ## Close the session
            #------------------------------------------------------------------------------------------------------------------------
            @reactive.Effect
            @reactive.event(input.exit)
            async def _():
                await session.close()
            
        self.app_ui = app_ui
        self.app_server = server