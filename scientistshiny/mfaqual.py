# -*- coding: utf-8 -*-
from shiny import Inputs, Outputs, Session, render, ui, reactive
import shinyswatch
import numpy as np
import pandas as pd
import scipy as sp
import plotnine as pn
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from scientisttools import fviz_mfa_ind,fviz_mfa_mod,fviz_corrcircle,fviz_mfa_group,fviz_mfa_axes,fviz_eig,fviz_contrib,fviz_cos2,dimdesc
from scientistshiny.base import Base
from scientistshiny.function import *

class MFAQUALshiny(Base):
    """
    Multiple Factor Analysis for qualitatives variables (MFAQUAL) with scientistshiny
    ---------------------------------------------------------------------------------

    Description
    -----------
    Performs Multiple Factor Analysis for qualitatives variables (MFAQUAL) with supplementary individuals and supplementary groups of quantitatives/qualitatives/mixed variables on a Shiny application. Allows to change MFA parameters and graphical parmeters. Graphics can be downloaded in png, jpg and pdf.

    Usage
    -----
    ```python
    >>> MFAQUALshiny(model)
    ```

    Parameters
    ----------
    `model`: an object of class MFAQUAL. A MFAQUAL result from scientisttools.

    Returns
    -------
    `Graphs` : a tab containing the individuals factor map and the variables factor map

    `Values` : a tab containing the the eigenvalue, the results for the variables, the results for the individuals, the results for the supplementary variables and the results for the categorical variables.

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
    
    ```
    
    for jupyter notebooks
    https://stackoverflow.com/questions/74070505/how-to-run-fastapi-application-inside-jupyter
    """
    def __init__(self,model=None):
        # Check if model is Multiple Factor Analysis for qualitatives variables (MFAQUAL)
        if model.model_ != "mfaqual":
            raise TypeError("'model' must be an object of class MFAQUAL")
        
        graph_choice = {"fviz_ind":"Individus","fviz_var":"Variables qualitatives","fviz_group": "Groupes","fviz_axes":"Axes partiels"}
        
        # Individuals test color
        ind_text_color_choices = {"actif/sup":"actifs/supplémentaires","cos2":"Cosinus","contrib":"Contribution","kmeans":"KMeans"}

        # resume choice
        resume_choices = {"stats_desc":"Statistiques descriptives","bar_plot":"Diagramme en barres", "chi2_test" : "Test du Chi2","others_test" : "Autres mesures d'association"}
        
        # Qualitatives variables columns
        var_labels = model.call_["X"].columns.tolist()
            
        # Initialise value choice
        value_choice = {"eigen_res":"Valeurs propres","quali_var_res":"Résultats pour les variables qualitatives","ind_res":"Résultats pour les individus","group_res" : "Resultats pour les groupes","axes_res": "Résultats pour les axes partiels"}

        # Check if supplementary individuals
        if hasattr(model, "ind_sup_"):
            value_choice.update({"ind_sup_res" : "Résultats pour les individus supplémentaires"})
        
         # Check if supplementary qualitatives variables
        if hasattr(model, "quali_var_sup_"):
            ind_text_color_choices.update({"var_qual":"Variable qualitative"})
            value_choice.update({"quali_var_sup_res" : "Résultats pour les variables qualitatives supplémentaires"})
            var_labels = [*var_labels,*model.quali_var_sup_["eta2"].index.tolist()]

        # Check if supplementary quantitatives variables
        if hasattr(model, "quanti_var_sup_"):
            graph_choice.update({"fviz_quanti_var_sup" : "Variables quantitatives supplémentaires"})
            value_choice.update({"quanti_var_sup_res" : "Résultats pour les variables quantitatives supplémentaires"})
            ind_text_color_choices.update({"var_quant":"Variable quantitative"})
            resume_choices.update({"hist_plot" : "Histogramme"})
        
        # Add supplementary group radio buttons
        if model.num_group_sup is not None:
            value_choice.update({"group_sup_res":"Résultats pour les groupes supplémentaires"})
            
        # UI
        app_ui = ui.page_fluid(
            ui.include_css(css_path),
            shinyswatch.theme.superhero(),
            header(title="Analyse Factorielle Multiple pour variables qualitatives",model_name="MFAQUAL"),
            ui.page_sidebar(
                ui.sidebar(
                    ui.panel_well(
                        ui.h6("Options graphiques",style="text-align:center"),
                        ui.div(ui.h6("Axes"),style="display: inline-block;padding: 5px"),
                        axes_input_select(model=model),
                        ui.br(),
                        ui.div(ui.input_select(id="fviz_choice",label="Quel graphe voule-vous modifier?",choices=graph_choice,selected="fviz_ind",multiple=False,width="100%")),
                        ui.panel_conditional("input.fviz_choice ==='fviz_ind'",
                            title_input(id="ind_title",value="Individuals - MFAQUAL"),
                            text_size_input(which="ind"),
                            point_select_input(id="ind_point_select"),
                            ui.panel_conditional("input.ind_point_select === 'cos2'",ui.div(lim_cos2(id="ind_lim_cos2"),align="center")),
                            ui.panel_conditional("input.ind_point_select === 'contrib'",ui.div(lim_contrib(id="ind_lim_contrib"),align="center")              ),
                            text_color_input(id="ind_text_color",choices=ind_text_color_choices),
                            ui.panel_conditional("input.ind_text_color === 'actif/sup'",
                                ui.input_select(id="ind_text_actif_color",label="Individus actifs",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                                ui.output_ui("ind_text_sup"),
                                ui.output_ui("ind_text_quali_var_sup")
                            ),
                            ui.panel_conditional("input.ind_text_color === 'kmeans'",ui.input_numeric(id="ind_text_kmeans_nb_clusters",label="Choix du nombre de clusters",value=2,min=1,max=model.ind_["coord"].shape[0],step=1,width="100%")),
                            ui.panel_conditional("input.ind_text_color === 'var_qual'",ui.output_ui("ind_text_var_qual")),
                            ui.panel_conditional("input.ind_text_color === 'var_quant'",ui.output_ui("ind_text_var_quant")),
                            ui.input_switch(id="ind_plot_repel",label="repel",value=True)
                        ),
                        ui.panel_conditional("input.fviz_choice ==='fviz_var'",
                            title_input(id="var_title",value="Variables categories - MFAQUAL"),
                            text_size_input(which="var"),
                            point_select_input(id="var_point"),
                            ui.panel_conditional("input.var_point === 'cos2'",ui.div(lim_cos2(id="var_lim_cos2"),align="center")              ),
                            ui.panel_conditional("input.var_point === 'contrib'",ui.div(lim_contrib(id="var_lim_contrib"),align="center")              ),
                            ui.input_select(id="var_text_color",label="Colorier les points par :",choices={"actif/sup":"actives/supplémentaires","cos2":"Cosinus","contrib":"Contribution","group":"Groupes","kmeans":"KMeans"},selected="group",multiple=False,width="100%"),
                            ui.panel_conditional("input.var_text_color === 'actif/sup'",
                                ui.input_select(id="var_text_actif_color",label="Modalités actives",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                                ui.output_ui("var_text_sup")
                            ),
                            ui.panel_conditional("input.var_text_color === 'kmeans'",ui.input_numeric(id="var_text_kmeans_nb_clusters",label="Choix du nombre de clusters",value=2,min=1,max=model.quali_var_["coord"].shape[0],step=1,width="100%")),
                            ui.input_switch(id="var_plot_repel",label="repel",value=True)
                        ),
                        ui.panel_conditional("input.fviz_choice === 'fviz_group'",
                            title_input(id="group_title",value="Graphe des groupes - MFAQUAL"),
                            text_size_input(which="group"),
                            ui.input_select(id="group_text_color",label="Colorier les points par :",choices={"actif/sup":"actifs/supplémentaires","cos2":"Cosinus","contrib":"Contribution","kmeans":"KMeans"},selected="actif/sup",multiple=False,width="100%"),
                            ui.panel_conditional("input.group_text_color === 'actif/sup'",
                                ui.input_select(id="group_text_actif_color",label="Groupes actifs",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                                ui.output_ui("group_text_sup")
                            ),
                            ui.panel_conditional("input.group_text_color === 'kmeans'",ui.input_numeric(id="group_text_kmeans_nb_clusters",label="Choix du nombre de clusters",value=2,min=1,max=model.group_["coord"].shape[0],step=1,width="100%")),
                            ui.input_switch(id="group_plot_repel",label="repel",value=True)
                        ),
                        ui.panel_conditional("input.fviz_choice === 'fviz_axes'",
                            title_input(id="axes_title",value="Graphe des axes partiels - MFAQUAL"),
                            text_size_input(which="axes"),
                            ui.input_select(id="axes_text_color",label="Colorier les points par :",choices={"actif/sup":"actifs/supplémentaires","group":"Groupes"},selected="group",multiple=False,width="100%"),
                            ui.panel_conditional("input.axes_text_color ==='actif/sup'",ui.input_select(id="axes_text_actif_color",label="Axes actifs/supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"))
                        ),
                        ui.output_ui("quanti_var_sup_fviz")
                    ),
                    ui.div(ui.input_action_button(id="exit",label="Quitter l'application",style='padding:5px; background-color: #fcac44;text-align:center;white-space: normal;'),align="center"),
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
                                ui.div(ui.output_plot("fviz_var_plot",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                                ui.div(ui.download_button(id="download_var_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_var_plot_png",label="png",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_var_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
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
                        ),
                        ui.output_ui("quanti_var_sup_plot")
                    ),
                    ui.nav_panel("Valeurs",
                        ui.input_radio_buttons(id="value_choice",label=ui.h6("Quelles sorties voulez-vous?"),choices=value_choice,inline=True),
                        ui.br(),
                        eigen_panel(),
                        ui.panel_conditional("input.value_choice === 'quali_var_res'",
                            ui.input_radio_buttons(id="var_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","contrib":"Contributions","cos2":"Cos2 - Qualité de la représentation","vtest":"Valeur-test","eta2":"Eta2 - Rapport de corrélation","coord_partial":"Partial coordinates"},selected="coord",width="100%",inline=True),
                            ui.panel_conditional("input.var_choice === 'coord'",panel_conditional1(text="quali_var",name="coord")),
                            ui.panel_conditional("input.var_choice === 'contrib'",panel_conditional2(text="quali_var",name="contrib")),
                            ui.panel_conditional("input.var_choice === 'cos2'",panel_conditional2(text="quali_var",name="cos2")),
                            ui.panel_conditional("input.var_choice === 'vtest'",panel_conditional1(text="quali_var",name="vtest")),
                            ui.panel_conditional("input.var_choice === 'eta2'",panel_conditional1(text="quali_var",name="eta2")),
                            ui.panel_conditional("input.var_choice === 'coord_partial'",panel_conditional1(text="quali_var",name="coord_partial")),
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
                        ui.output_ui("quali_var_sup_panel"),
                        ui.output_ui("quanti_var_sup_panel"),
                        ui.output_ui("group_sup_panel")
                    ),
                    dim_desc_panel(model=model),
                    ui.nav_panel("Résumé du jeu de données",
                        ui.input_radio_buttons(id="resume_choice",label=ui.h6("Que voulez - vous afficher?"),choices=resume_choices,selected="bar_plot",width="100%",inline=True),
                        ui.br(),
                        ui.panel_conditional("input.resume_choice === 'bar_plot'",
                            ui.row(
                                ui.column(2,
                                    ui.input_select(id="var_qual_label",label=ui.h6("Choisir une variable"),choices={x:x for x in var_labels},selected=var_labels[0])
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
                        ),
                        ui.panel_conditional("input.resume_choice === 'chi2_test'",panel_conditional1(text="chi2",name="test")),
                        ui.panel_conditional("input.resume_choice === 'others_test'",panel_conditional1(text="others",name="test")),
                        ui.output_ui("quanti_sup_graph")
                    ),
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
            
            if hasattr(model,"quali_var_sup_"):
                @render.ui
                def ind_text_quali_var_sup():
                    return ui.TagList(ui.input_select(id="ind_text_quali_var_sup_color",label="Modalités supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="red",multiple=False,width="100%"))
           
            #------------------------------------------------------------------------------------------------------------------
            # Disable individuals colors
            if hasattr(model,"ind_sup_") and hasattr(model,"quali_var_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_actif_color",label="Individus actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.ind_text_sup_color(),input.ind_text_quali_var_sup_color()]]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_sup_color",label="Individus supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.ind_text_actif_color(),input.ind_text_quali_var_sup_color()]]},selected="blue")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_quali_var_sup_color",label="Modalités supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.ind_text_actif_color(),input.ind_text_sup_color()]]},selected="red")
            elif hasattr(model,"ind_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_actif_color",label="Individus actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.ind_text_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_sup_color",label="Individus supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.ind_text_actif_color()]},selected="blue")
            elif hasattr(model,"quali_var_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_actif_color",label="Individus actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.ind_text_quali_var_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="ind_text_quali_var_sup_color",label="Modalités supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.ind_text_actif_color()]},selected="red")

            #-------------------------------------------------------------------------------------------------
            if hasattr(model,"quali_var_sup_"):
                @render.ui
                def ind_text_var_qual():
                    quali_var_sup_labels = model.quali_var_sup_["eta2"].index.tolist()
                    return ui.TagList(
                            ui.input_select(id="ind_text_var_qual_color",label="Choix de la variable",choices={x:x for x in quali_var_sup_labels},selected=quali_var_sup_labels[0],multiple=False,width="100%"),
                            ui.input_switch(id="ind_text_add_ellipse",label="Trace les ellipses de confiance autour des modalités",value=False)
                        )

            #-------------------------------------------------------------------------------------------------
            if hasattr(model,"quanti_var_sup_"):
                @render.ui
                def ind_text_var_quant():
                    quanti_sup_labels = model.quanti_var_sup_["coord"].index.tolist()
                    return ui.TagList(ui.input_select(id="ind_text_var_quant_color",label="Choix de la variable",choices={x:x for x in quanti_sup_labels},selected=quanti_sup_labels[0],multiple=False,width="100%"))
            
            #-----------------------------------------------------------------------------------------------
            if hasattr(model,"quali_var_sup_"):
                @render.ui
                def var_text_sup():
                    return ui.TagList(ui.input_select(id="var_text_sup_color",label="Modalités supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="blue",multiple=False,width="100%"))
            
                # Disable qualitative variable colors
                @reactive.Effect
                def _():
                    ui.update_select(id="var_text_actif_color",label="Modalités actives",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.var_text_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="var_text_sup_color",label="Modalités supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.var_text_actif_color()]},selected="blue")
            
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
            ## Individuals - MFAQUAL
            #---------------------------------------------------------------------------------
            @reactive.Calc
            def plot_ind():
                # Define boolean
                if hasattr(model,"ind_sup_"):
                    ind_sup = True
                else:
                    ind_sup = False
                    
                if hasattr(model,"quali_var_sup_"):
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
                    if hasattr(model,"quali_var_sup_"):
                        color_quali_sup = input.ind_text_quali_var_sup_color()
                    else:
                        color_quali_sup = None

                    fig = fviz_mfa_ind(self=model,
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
                                       repel=input.ind_plot_repel())
                elif input.ind_text_color() in ["cos2","contrib"]:
                    fig = fviz_mfa_ind(self=model,
                                       axis=[int(input.axis1()),int(input.axis2())],
                                       color=input.ind_text_color(),
                                       ind_sup=ind_sup,
                                       quali_sup=quali_sup,
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
                                       quali_sup = quali_sup,
                                       text_size = input.ind_text_size(),
                                       lim_contrib = input.ind_lim_contrib(),
                                       lim_cos2 = input.ind_lim_cos2(),
                                       title = input.ind_title(),
                                       repel=input.ind_plot_repel())
                elif input.ind_text_color() == "var_quant":
                    fig = fviz_mfa_ind(self = model,
                                       axis = [int(input.axis1()),int(input.axis2())],
                                       color = input.ind_text_var_quant_color(),
                                       ind_sup = ind_sup,
                                       quali_sup = quali_sup,
                                       text_size = input.ind_text_size(),
                                       lim_contrib = input.ind_lim_contrib(),
                                       lim_cos2 = input.ind_lim_cos2(),
                                       title = input.ind_title(),
                                       legend_title=input.ind_text_var_quant_color(),
                                       repel=input.ind_plot_repel())
                elif input.ind_text_color() == "var_qual":
                    fig = fviz_mfa_ind(self = model,
                                       axis = [int(input.axis1()),int(input.axis2())],
                                       text_size = input.ind_text_size(),
                                       ind_sup = ind_sup,
                                       quali_sup = quali_sup,
                                       lim_contrib = input.ind_lim_contrib(),
                                       lim_cos2 = input.ind_lim_cos2(),
                                       title = input.ind_title(),
                                       habillage= input.ind_text_var_qual_color(),
                                       add_ellipse=input.ind_text_add_ellipse(),
                                       repel=input.ind_plot_repel())
                return fig+pn.theme_gray()

            # ------------------------------------------------------------------------------
            # Individual - MFAQUAL
            @render.plot(alt="Individuals - MFAQUAL")
            def fviz_ind_plot():
                return plot_ind().draw()
            
            # Downlaod
            # @render.download(filename="Individuals-Factor-Map.png")
            # def download_ind_plot_png():
            #    return plot_ind().save("Individuals-Factor-Map.png")
            
            #--------------------------------------------------------------------------------------------
            ## Variables categories - MFAQUAL
            #--------------------------------------------------------------------------------------------
            @reactive.Calc
            def plot_var():
                if hasattr(model, "quali_var_sup_"):
                    quali_sup = True
                else:
                    quali_sup = False
                
                if input.var_text_color() == "actif/sup":
                    if hasattr(model, "quali_var_sup_"):
                        color_sup = input.var_text_sup_color()
                    else:
                        color_sup = None
                    
                    fig = fviz_mfa_mod(self=model,
                                       axis=[int(input.axis1()),int(input.axis2())],
                                       title=input.var_title(),
                                       color=input.var_text_actif_color(),
                                       quali_sup=quali_sup,
                                       color_sup=color_sup,
                                       text_size=input.var_text_size(),
                                       lim_contrib = input.var_lim_contrib(),
                                       lim_cos2 = input.var_lim_cos2(),
                                       repel=input.var_plot_repel())
                elif input.var_text_color() in ["cos2","contrib","group"]:
                    fig = fviz_mfa_mod(self=model,
                                       axis=[int(input.axis1()),int(input.axis2())],
                                       title=input.var_title(),
                                       color=input.var_text_color(),
                                       quali_sup=quali_sup,
                                       text_size=input.var_text_size(),
                                       lim_contrib = input.var_lim_contrib(),
                                       lim_cos2 = input.var_lim_cos2(),
                                       repel=input.var_plot_repel())
                elif input.var_text_color() == "kmeans":
                    kmeans = KMeans(n_clusters=input.var_text_kmeans_nb_clusters(), random_state=np.random.seed(123), n_init="auto").fit(model.quali_var_["coord"])
                    fig = fviz_mfa_mod(self=model,
                                       axis=[int(input.axis1()),int(input.axis2())],
                                       title=input.var_title(),
                                       color=kmeans,
                                       quali_sup=quali_sup,
                                       text_size=input.var_text_size(),
                                       lim_contrib = input.var_lim_contrib(),
                                       lim_cos2 = input.var_lim_cos2(),
                                       repel=input.var_plot_repel())
                return fig+pn.theme_gray()

            @render.plot(alt="Variables categories - MFAQUAL")
            def fviz_var_plot():
                return plot_var().draw()
            
            #-----------------------------------------------------------------
            ## Groups Factor Map
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
            
            @render.plot(alt="Groups Factor Map - MFAQUAL")
            def fviz_group_plot():
                return plot_group().draw()
            
            #----------------------------------------------------------------------------------
            ## Axes Partiels Factor Map
            #----------------------------------------------------------------------------------
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
            
            @render.plot(alt="Axes partiels Factor Map - MFAQUAL")
            def fviz_axes_plot():
                return plot_axes().draw()
            
            #------------------------------------------------------------------------------------------
            ## Supplementary quantitatives variables correlation circle
            #------------------------------------------------------------------------------------------
            if hasattr(model,"quanti_var_sup_"):
                @render.ui
                def quanti_var_sup_fviz():
                    return ui.panel_conditional("input.fviz_choice ==='fviz_quanti_var_sup'",
                            title_input(id="quanti_var_sup_title",value="Correlation Circle - MFAQUAL"),
                            text_size_input(which="quanti_var_sup"),
                            ui.input_select(id="quanti_var_sup_color",label="Variables quantitatives supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="black")
                        )

                @render.ui
                def quanti_var_sup_plot():
                    return ui.TagList(
                        ui.br(),
                        ui.row(
                            ui.column(5,
                                ui.div(ui.output_plot("fviz_quanti_var_sup_plot",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px"),
                                ui.div(ui.download_button(id="download_quanti_var_sup_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_quanti_var_sup_plot_png",label="png",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_quanti_var_sup_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                align="center"
                            )
                        ),
                    )
                
                # Correlation circle Factor Map
                @reactive.Calc
                def plot_circle():
                    fig = fviz_corrcircle(self=model,axis=[int(input.axis1()),int(input.axis2())],title=input.quanti_var_sup_title(),text_size=input.quanti_var_sup_text_size(),color=input.quanti_var_sup_color(),ggtheme=pn.theme_gray())
                    return fig
                
                @render.plot(alt="Correlation Circle - MFAQUAL")
                def fviz_quanti_var_sup_plot():
                    return plot_circle().draw()

            #-------------------------------------------------------------------------------------------
            ## Eigenvalue informations
            #-------------------------------------------------------------------------------------------
            # Eigenvalue - Scree plot
            @render.plot(alt="Scree Plot - MFAQUAL")
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
            ## Qualitatives variables informations
            #----------------------------------------------------------------------------------------------------------------------------------------------
            # Qualitatives variables Coordinates
            @render.data_frame
            def quali_var_coord_table():
                quali_var_coord = model.quali_var_["coord"].round(4).reset_index()
                quali_var_coord.columns = ["Categories", *quali_var_coord.columns[1:]]
                return DataTable(data=match_datalength(data=quali_var_coord,value=input.quali_var_coord_len()),filters=input.quali_var_coord_filter())
            
            # Qualitatives variables Contributions
            @render.data_frame
            def quali_var_contrib_table():
                quali_var_contrib = model.quali_var_["contrib"].round(4).reset_index()
                quali_var_contrib.columns = ["Categories", *quali_var_contrib.columns[1:]]
                return  DataTable(data=match_datalength(data=quali_var_contrib,value=input.quali_var_contrib_len()),filters=input.quali_var_contrib_filter())
            
            # Add Qualitative variables Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.quali_var_contrib_graph_btn)
            def _():
                graph_modal_show(text="quali_var",name="contrib",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def quali_var_contrib_plot():
                fig = fviz_contrib(self=model,choice="quali_var",axis=input.quali_var_contrib_axis(),top_contrib=int(input.quali_var_contrib_top()),color=input.quali_var_contrib_color(),bar_width=input.quali_var_contrib_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot variables Contributions
            @render.plot(alt="Qualitative variables Contributions Map - MFAQUAL")
            def fviz_quali_var_contrib():
                return quali_var_contrib_plot().draw()
            
            # Qualitative Variables Cos2 
            @render.data_frame
            def quali_var_cos2_table():
                quali_var_cos2 = model.quali_var_["cos2"].round(4).reset_index()
                quali_var_cos2.columns = ["Categories", *quali_var_cos2.columns[1:]]
                return  DataTable(data=match_datalength(data=quali_var_cos2,value=input.quali_var_cos2_len()),filters=input.quali_var_cos2_filter())
            
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.quali_var_cos2_graph_btn)
            def _():
                graph_modal_show(text="quali_var",name="cos2",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def quali_var_cos2_plot():
                fig = fviz_cos2(self=model,choice="quali_var",axis=input.quali_var_cos2_axis(),top_cos2=int(input.quali_var_cos2_top()),color=input.quali_var_cos2_color(),bar_width=input.quali_var_cos2_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot variables Cos2
            @render.plot(alt="Qualitative variables cosines Map - MFAQUAL")
            def fviz_quali_var_cos2():
                return quali_var_cos2_plot().draw()
            
            # Qualitative variables value - test
            @render.data_frame
            def quali_var_vtest_table():
                quali_var_vtest = model.quali_var_["vtest"].round(4).reset_index()
                quali_var_vtest.columns = ["Categories", *quali_var_vtest.columns[1:]]
                return DataTable(data=match_datalength(data=quali_var_vtest,value=input.quali_var_vtest_len()),filters=input.quali_var_vtest_filter())
            
            # Qualitative variables square correlation ratio
            @render.data_frame
            def quali_var_eta2_table():
                quali_var_eta2 = model.quali_var_["eta2"].round(4).reset_index()
                quali_var_eta2.columns = ["Variables", *quali_var_eta2.columns[1:]]
                return DataTable(data=match_datalength(data=quali_var_eta2,value=input.quali_var_eta2_len()),filters=input.quali_var_eta2_filter())
            
            # Partiel coordinates
            @render.data_frame
            def quali_var_coord_partial_table():
                quali_var_coord_partial = reset_columns(X=model.quali_var_["coord_partiel"].round(4)).reset_index()
                quali_var_coord_partial.columns = ["Categories", *quali_var_coord_partial.columns[1:]]
                return  DataTable(data = match_datalength(quali_var_coord_partial,input.quali_var_coord_partial_len()),filters=input.quali_var_coord_partial_filter())

            #------------------------------------------------------------------------------------------
            # Supplementary qualitative variables
            ##-----------------------------------------------------------------------------------------
            if hasattr(model,"quali_var_sup_"):
                @render.ui
                def quali_var_sup_panel():
                    return ui.panel_conditional("input.value_choice == 'quali_var_sup_res'",
                                ui.input_radio_buttons(id="quali_var_sup_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","cos2":"Cos2 - Qualité de la représentation","vtest":"Valeur-test","eta2" : "Eta2 - Rapport de corrélation","coord_partial": "Partial coordinates"},selected="coord",width="100%",inline=True),
                                ui.panel_conditional("input.quali_var_sup_choice === 'coord'",panel_conditional1(text="quali_var_sup",name="coord")),
                                ui.panel_conditional("input.quali_var_sup_choice === 'cos2'",panel_conditional1(text="quali_var_sup",name="cos2")),
                                ui.panel_conditional("input.quali_var_sup_choice === 'vtest'",panel_conditional1(text="quali_var_sup",name="vtest")),
                                ui.panel_conditional("input.quali_var_sup_choice === 'eta2'",panel_conditional1(text="quali_var_sup",name="eta2")),
                                ui.panel_conditional("input.quali_var_sup_choice === 'coord_partial'",panel_conditional1(text="quali_var_sup",name="coord_partial"))
                            )
                
                # Factor coordinates
                @render.data_frame
                def quali_var_sup_coord_table():
                    quali_var_sup_coord = model.quali_var_sup_["coord"].round(4).reset_index()
                    quali_var_sup_coord.columns = ["Categories", *quali_var_sup_coord.columns[1:]]
                    return  DataTable(data = match_datalength(quali_var_sup_coord,input.quali_var_sup_coord_len()),filters=input.quali_var_sup_coord_filter())
                
                # Square cosinus
                @render.data_frame
                def quali_var_sup_cos2_table():
                    quali_var_sup_cos2 = model.quali_var_sup_["cos2"].round(4).reset_index()
                    quali_var_sup_cos2.columns = ["Categories", *quali_var_sup_cos2.columns[1:]]
                    return  DataTable(data = match_datalength(quali_var_sup_cos2,input.quali_var_sup_cos2_len()),filters=input.quali_var_sup_cos2_filter())
                
                # Value - test
                @render.data_frame
                def quali_var_sup_vtest_table():
                    quali_var_sup_vtest = model.quali_var_sup_["vtest"].round(4).reset_index()
                    quali_var_sup_vtest.columns = ["Categories", *quali_var_sup_vtest.columns[1:]]
                    return  DataTable(data = match_datalength(quali_var_sup_vtest,input.quali_var_sup_vtest_len()),filters=input.quali_var_sup_vtest_filter())
                
                # Square correlation ratio
                @render.data_frame
                def quali_var_sup_eta2_table():
                    quali_var_sup_eta2 = model.quali_var_sup_["eta2"].round(4).reset_index()
                    quali_var_sup_eta2.columns = ["Variables", *quali_var_sup_eta2.columns[1:]]
                    return  DataTable(data = match_datalength(quali_var_sup_eta2,input.quali_var_sup_eta2_len()),filters=input.quali_var_sup_eta2_filter())
                
                # Partiel coordinates
                @render.data_frame
                def quali_var_sup_coord_partial_table():
                    quali_var_sup_coord_partial = reset_columns(X=model.quali_var_sup_["coord_partiel"].round(4)).reset_index()
                    quali_var_sup_coord_partial.columns = ["Categories", *quali_var_sup_coord_partial.columns[1:]]
                    return  DataTable(data = match_datalength(quali_var_sup_coord_partial,input.quali_var_sup_coord_partial_len()),filters=input.quali_var_sup_coord_partial_filter())

            #---------------------------------------------------------------------------------
            ## Supplementary quantitative Variables
            #---------------------------------------------------------------------------------
            if hasattr(model,"quanti_var_sup_"):
                @render.ui
                def quanti_var_sup_panel():
                    return ui.panel_conditional("input.value_choice == 'quanti_var_sup_res'",
                                ui.input_radio_buttons(id="quanti_var_sup_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                                ui.panel_conditional("input.quanti_var_sup_choice === 'coord'",panel_conditional1(text="quanti_var_sup",name="coord")),
                                ui.panel_conditional("input.quanti_var_sup_choice === 'cos2'",panel_conditional1(text="quanti_var_sup",name="cos2"))
                            )
                
                # Factor Coordinates
                @render.data_frame
                def quanti_var_sup_coord_table():
                    quanti_var_sup_coord = model.quanti_var_sup_["coord"].round(4).reset_index()
                    quanti_var_sup_coord.columns = ["Variables", *quanti_var_sup_coord.columns[1:]]
                    return DataTable(data=match_datalength(data=quanti_var_sup_coord,value=input.quanti_var_sup_coord_len()),filters=input.quanti_var_sup_coord_filter())
                
                # Square cosinus
                @render.data_frame
                def quanti_var_sup_cos2_table():
                    quanti_var_sup_cos2 = model.quanti_var_sup_["cos2"].round(4).reset_index()
                    quanti_var_sup_cos2.columns = ["Variables", *quanti_var_sup_cos2.columns[1:]]
                    return DataTable(data=match_datalength(data=quanti_var_sup_cos2,value=input.quanti_var_sup_cos2_len()),filters=input.quanti_var_sup_cos2_filter())
            
            #--------------------------------------------------------------------------------------------------------
            # Individuals informations
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
            @render.plot(alt="Individuals Contributions Map - MFAQUAL")
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
            @render.plot(alt="Individuals Cosines Map - MFAQUAL")
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

            #--------------------------------------------------------------------------------------------
            ## Supplementary individuals
            #---------------------------------------------------------------------------------------------
            if hasattr(model,"ind_sup_"):
                @render.ui
                def ind_sup_panel():
                    return ui.panel_conditional("input.value_choice == 'ind_sup_res'",
                                ui.input_radio_buttons(id="ind_sup_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","cos2":"Cos2 - Qualité de la représentation","coord_partiel": "Partial coordinates"},selected="coord",width="100%",inline=True),
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
            @render.plot(alt="Group Contributions Map - MFAQUAL")
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
            @render.plot(alt="Group Cosines Map - MFAQUAL")
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
            @render.plot(alt="Axes partiels Contributions Map - MFAQUAL")
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
            #--------------------------------------------------------------------------------------------------
            @reactive.Calc
            def data():
                data = model.call_["Xtot"].loc[:,var_labels].astype("object")
                if model.ind_sup is not None:
                    data = data.drop(index=model.call_["ind_sup"])
                return data
            
            # Descriptive statistics
            @render.data_frame
            def stats_desc_table():
                stats_desc = data().describe(include="all").round(4).T.reset_index().rename(columns={"index":"Variables"})
                return  DataTable(data = match_datalength(stats_desc,input.stats_desc_len()),filters=input.stats_desc_filter())

            @reactive.Calc
            def plot_bar():
                return pn.ggplot(data(),pn.aes(x=input.var_qual_label()))+ pn.geom_bar(color="black",fill="gray")
            
            # Diagramme en barres
            @render.plot(alt="Bar-Plot")
            def fviz_bar_plot():
                return plot_bar().draw()
            
            # Chi-square test
            @render.data_frame
            def chi2_test_table():
                chi2_test = pd.DataFrame(columns=["variable1","variable2","statistic","dof","pvalue"]).astype("float")
                idx = 0
                for i in np.arange(data().shape[1]-1):
                    for j in np.arange(i+1,data().shape[1]):
                        tab = pd.crosstab(data().iloc[:,i],data().iloc[:,j])
                        statistic, pvalue, dof, _ = sp.stats.chi2_contingency(observed = tab,correction=False)
                        row_others = pd.DataFrame({"variable1" : data().columns[i],
                                                   "variable2" : data().columns[j],
                                                   "statistic"    : round(statistic,4),
                                                   "dof" : int(dof),
                                                   "pvalue"   : round(pvalue,4)},
                                                   index=[idx])
                        chi2_test = pd.concat((chi2_test,row_others),axis=0,ignore_index=True)
                        idx = idx + 1
                return  DataTable(data = match_datalength(chi2_test,input.chi2_test_len()),filters=input.chi2_test_filter())
            
            @render.data_frame
            def others_test_table():
                others_test = pd.DataFrame(columns=["variable1","variable2","cramer","tschuprow","pearson"]).astype("float")
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
            
            # Add supplementary quantitatives variables summary
            if hasattr(model,"quanti_var_sup_"):
                quanti_sup_var_labels = model.quanti_var_sup_["coord"].index
                quanti_data = model.call_["Xtot"].loc[:,quanti_sup_var_labels]
                if hasattr(model,"ind_sup_"):
                    quanti_data = quanti_data.drop(index=model.call_["ind_sup"])
                
                @render.ui
                def quanti_sup_graph():
                    return ui.TagList(
                        ui.panel_conditional("input.resume_choice === 'stats_desc'",panel_conditional1(text="stats",name="desc")),
                            ui.panel_conditional("input.resume_choice === 'hist_plot'",
                                ui.row(
                                    ui.column(2,
                                        ui.input_select(id="quanti_var_sup_label",label=ui.h6("Choisir une variable"),choices={x:x for x in quanti_sup_var_labels},selected=quanti_sup_var_labels[0]),
                                        ui.input_switch(id="add_density",label="Densite",value=False)
                                    ),
                                    ui.column(10,ui.div(ui.output_plot(id="fviz_hist_plot",width='100%',height='500px'),align="center"))
                                )
                        )
                    )

                # Histogram for quantitatives variables
                @reactive.Calc
                def plot_hist():
                    p = pn.ggplot(quanti_data,pn.aes(x=input.quanti_var_sup_label()))
                    # Add density
                    if input.add_density():
                        p = p + pn.geom_histogram(pn.aes(y="..density.."), color="black", fill="gray")+pn.geom_density(alpha=.2, fill="#FF6666")
                    else:
                        p = p + pn.geom_histogram(color="black", fill="gray")
                    return p + pn.ggtitle(f"Histogram de {input.quanti_var_sup_label()}")

                @render.plot(alt="Histogram - MFAQUAL")
                def fviz_hist_plot():
                    return plot_hist().draw()

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