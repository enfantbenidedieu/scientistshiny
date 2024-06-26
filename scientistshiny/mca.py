# -*- coding: utf-8 -*-
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
import shinyswatch
from pathlib import Path
import numpy as np
import scipy as sp
import pandas as pd
import plotnine as pn
import matplotlib.colors as mcolors
import nest_asyncio
import uvicorn

#  All scientisttools fonctions
from scientisttools import fviz_mca_ind,fviz_mca_mod,fviz_mca_var,fviz_eig, fviz_contrib,fviz_cos2,fviz_corrplot,fviz_corrcircle, dimdesc

from .function import *

colors = mcolors.CSS4_COLORS
colors["cos2"] = "cos2"
colors["contrib"] = "contrib"

css_path = Path(__file__).parent / "www" / "style.css"

class MCAshiny:
    """
    Multiple Correspondance Analysis (MCA) with scientistshiny
    ----------------------------------------------------------

    Description
    -----------
    Performs Multiple Correspondance Analysis (MCA) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables on a Shiny application.
    Graphics can be downloaded in png, jpg and pdf.

    Usage
    -----
    MCAshiny(fa_model)

    Parameters:
    ----------
    model : An instance of class MCA. A MCA result from scientisttools.

    Returns:
    -------
    Graphs : a tab containing the individuals factor map and the variables factor map

    Values : a tab containing the summary of the MCA performed, the eigenvalue, the results
             for the variables, the results for the individuals, the results for the supplementary
             variables and the results for the categorical variables.

    Automatic description of axes : a tab containing the output of the dimdesc function. This function is designed to 
                                    point out the variables and the categories that are the most characteristic according
                                    to each dimension obtained by a Factor Analysis.

    Summary of dataset : A tab containing the summary of the dataset and a boxplot and histogramm for quantitative variables.

    Data : a tab containing the dataset with a nice display.

    The left part of the application allows to change some elements of the graphs (axes, variables, colors,.)

    Author:
    -------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples:
    ---------
    > from scientisttools import MCA
    > from scientistshiny import MCAshiny


    for jupyter notebooks
    https://stackoverflow.com/questions/74070505/how-to-run-fastapi-application-inside-jupyter
    """


    def __init__(self,model=None):
        # Check if model is Multiple Correspondence Analysis (CA)
        if model.model_ != "mca":
            raise TypeError("'model' must be an instance of class MCA")
        
        # -----------------------------------------------------------------------------------
        # Initialise value choice
        value_choice = {"EigenRes" : "Valeurs propres",
                        "ModRes"   : "Résultats des modalités",
                        "IndRes"   : "Résultats sur les individus",
                        "VarRes"   : "Résultats sur les variables"}
        
        # Check if supplementary individuals
        if hasattr(model,"row_sup_"):
            value_choice = {**value_choice,**{"IndSupRes" : "Résultats des individus supplémentaires"}}

        # Check if supplementary quantitatives variables
        if hasattr(model,"quanti_sup_"):
            value_choice = {**value_choice, **{"VarQuantRes" : "Résultats des variables quantitatives supplémentaires"}}

        # Check if supplementary qualitatives variables
        if hasattr(model,"quali_sup_"):
            value_choice = {**value_choice,**{"VarSupRes" : "Résultats des variables qualitatives supplémentaires"}}
        
        # Plot Choice
        PlotChoice = {"IndPlot":"individus",
                      "ModPlot":"modalités",
                      "VarPlot":"Variables"}
        if hasattr(model,"quanti_sup_"):
            PlotChoice = {**PlotChoice,**{"VarQuantPlot" : "Variables quantitatives"}}
        
        # Dim Desc Choice
        DimDescChoice = {}
        for i in range(min(3,model.call_["n_components"])):
            DimDescChoice.update({"Dim."+str(i+1) : "Dimension "+str(i+1)})
        
        # Add Supplementary Qualitatives Variables
        VarLabelChoice = model.call_["X"].columns.tolist()
        if hasattr(model,"quali_sup_"):
            VarLabelChoice = [*VarLabelChoice,*model.quali_sup_["eta2"].index.tolist()]

        # UI
        app_ui = ui.page_fluid(
            ui.include_css(css_path),
            shinyswatch.theme.superhero(),
            ui.page_navbar(
                title=ui.div(ui.panel_title(ui.h2("Analyse des Correspondances Multiples"),window_title="MCAshiny"),
                align="center"),
                inverse=True,id="navbar_id",padding={"style": "text-align: center;"}),
            ui.page_sidebar(
                ui.sidebar(
                    ui.panel_well(
                        ui.h6("Options graphiques",style="text-align:center"),
                    ui.div(ui.h6("Axes"),style="display: inline-block;padding: 5px"),
                    ui.div(
                        ui.input_select(
                            id="Axis1",
                            label="",
                            choices={x:x for x in range(model.call_["n_components"])},
                            selected=0,
                            multiple=False
                        ),
                        style="display: inline-block;"
                    ),
                    ui.div(
                        ui.input_select(
                            id="Axis2",
                            label="",
                            choices={x:x for x in range(model.call_["n_components"])},
                            selected=1,
                            multiple=False
                        ),
                        style="display: inline-block;"
                    ),
                    ui.br(),
                    ui.div(
                        ui.input_radio_buttons(
                            id="IndModVar",
                            label="Modifier le graphe des",
                            choices=PlotChoice,
                            selected="IndPlot",
                            inline=True,
                            width="100%"
                        ),
                        style="display: inline-block;"
                    ),
                    ui.panel_conditional("input.IndModVar ==='IndPlot'",
                        ui.input_text(
                            id="IndTitle",
                            label="Titre du graphe",
                            value="Individuals Factor Map - MCA",
                            width="100%"
                        ),
                        ui.input_slider(
                            id="IndTextSize",
                            label="Taille des libellés",
                            min=8,
                            max=20,
                            value=8,
                            step=2,
                            ticks=False
                        ),
                        ui.input_select(
                            id="IndPointSelect",
                            label="Libellés des points pour",
                            choices={
                                "none"    : "Pas de sélection",
                                "cos2"    : "Cosinus",
                                "contrib" : "Contribution"
                                },
                            selected="none",
                            multiple=False,
                            width="100%"
                        ),
                        ui.panel_conditional("input.IndPointSelect === 'cos2'",
                            ui.div(
                                ui.input_slider(
                                    id="IndLimCos2", 
                                    label = "Libellés pour un cos2 plus grand que",
                                    min = 0, 
                                    max = 1,
                                    value=0,
                                    step=0.05
                                ),
                                align="center"
                            )              
                        ),
                        ui.panel_conditional("input.IndPointSelect === 'contrib'",
                            ui.div(
                                ui.input_slider(
                                    id="IndLimContrib",
                                    label ="Libellés pour une contribution plus grande que",
                                    min = 0, 
                                    max = 100,
                                    value=0,
                                    step=5
                                ),
                                align="center"
                            )              
                        ),
                        ui.input_select(
                            id="IndTextColor",
                            label="Colorier les points par :",
                            choices={
                                "actif/sup": "actifs/supplémentaires",
                                "cos2"     : "Cosinus",
                                "contrib"  : "Contribution",
                                "varqual"  : "Variable qualitative",
                                "varquant" : "Variable quantitative"
                            },
                            selected="actif/sup",
                            multiple=False,
                            width="100%"
                        ),
                        ui.panel_conditional(
                            "input.IndTextColor==='actif/sup'",
                            ui.output_ui("IndTextChoice"),
                        ),
                        ui.panel_conditional(
                            "input.IndTextColor==='varqual'",
                             ui.input_select(
                                 id="IndTextVarQualColor",
                                 label="choix de la variable",
                                 choices={x:x for x in VarLabelChoice},
                                 selected=VarLabelChoice[0],
                                 multiple=False,
                                 width="100%"
                             ),
                             ui.input_switch(
                                 id="IndAddEllipse",
                                 label="Trace les ellipses de confiance autour des barycentres",
                                 value=False
                             )
                        ),
                        ui.panel_conditional(
                            "input.IndTextColor==='varquant'",
                            ui.output_ui("IndVarQuantColorPanel")
                        ),
                        ui.input_switch(
                            id="IndPlotRepel",
                            label="repel",
                            value=True
                        )
                    ),
                    ui.panel_conditional("input.IndModVar==='ModPlot'",
                        ui.input_text(
                            id="ModTitle",
                            label="Titre du graphe",
                            value="Variables categories - MCA",
                            width="100%"
                        ),
                        ui.input_slider(
                            id="ModTextSize",
                            label="Taille des libellés",
                            min=8,
                            max=20,
                            value=8,
                            step=2,
                            ticks=False
                        ),
                        ui.input_select(
                            id="ModPointSelect",
                            label="Libellés des points pour",
                            choices={
                                "none"    : "Pas de sélection",
                                "cos2"    : "Cosinus",
                                "contrib" : "Contribution"
                                },
                            selected="none",
                            multiple=False,
                            width="100%"
                        ),
                        ui.panel_conditional("input.ModPointSelect === 'cos2'",
                            ui.div(ui.input_slider(id="ModLimCos2", label = "Libellés pour un cos2 plus grand que",min = 0, max = 1,value=0,step=0.05),align="center")              
                        ),
                        ui.panel_conditional("input.ModPointSelect === 'contrib'",
                            ui.div(ui.input_slider(id="ModLimContrib", label ="Libellés pour une contribution plus grande que",min = 0, max = 100,value=0,step=5),align="center")              
                        ),
                        ui.input_select(
                            id="ModTextColor",
                            label="Colorier les points par :",
                            choices={
                                "actif/sup": "actifs/supplémentaires",
                                "cos2"     : "Cosinus",
                                "contrib"  : "Contribution"
                            },
                            selected="actif/sup",
                            multiple=False,
                            width="100%"
                        ),
                        ui.panel_conditional(
                            "input.ModTextColor==='actif/sup'",
                            ui.output_ui("ModTextChoice"),
                        ),
                        ui.input_switch(
                            id="ModPlotRepel",
                            label="repel",
                            value=True
                        )
                    ),
                    ui.panel_conditional("input.IndModVar ==='VarPlot'",
                        ui.input_text(
                                id="VarTitle",
                                label='Titre du graphe',
                                value="Variables - MCA",
                                width="100%"
                        ),
                        ui.input_slider(
                            id="VarTextSize",
                            label="Taille des libellés",
                            min=8,
                            max=20,
                            value=8,
                            step=2,
                            ticks=False
                        ),
                        ui.output_ui("VarPlotColorChoice"),
                        ui.input_switch(
                            id="VarPlotRepel",
                            label="repel",
                            value=True
                        )
                    ),
                    ui.output_ui("VarQuantOptions"),
                    ui.div(
                        ui.input_action_button(
                            id="exit",
                            label="Quitter l'application",
                            style='padding:5px; background-color: #fcac44;text-align:center;white-space: normal;'
                        ),
                        align="center"),
                    ),
                    width="25%"
                ),
                ui.navset_card_tab(
                    ui.nav_panel("Graphes",
                        ui.row(
                            ui.column(6,
                                ui.div(ui.output_plot("RowFactorMap",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px"),
                                ui.div(ui.download_button(id="IndGraphDownloadJpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="IndGraphDownloadPng",label="png",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="IndGraphDownloadPdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                align="center"
                            ),
                            ui.column(6,
                                ui.div(ui.output_plot("ModFactorMap",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                                ui.div(ui.download_button(id="ModGraphDownloadJpg",label="jpg",style = download_btn_style,icon=None),style="display: inline-block;",align="center"),
                                ui.div(ui.download_button(id="ModGraphDownloadPng",label="png",style = "background-color: #1C2951;"),style="display: inline-block;",align="center"),
                                ui.div(ui.download_button(id="ModGraphDownloadPdf",label="pdf",style = "background-color: #1C2951;"),style="display: inline-block;",align="center"),
                                align="center"
                            )
                        ),
                        ui.br(),
                        ui.row(
                            ui.column(6,
                                ui.div(ui.output_plot("VarFactorMap",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px"),
                                ui.div(ui.download_button(id="VarGraphDownloadJpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="VarGraphDownloadPng",label="png",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="VarGraphDownloadPdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                align="center"
                            ),
                            ui.column(6,
                                ui.output_ui("VarQuantOutPut"),
                                align="center"
                            )
                        ),
                    ),
                    ui.nav_panel("Valeurs",
                        ui.input_radio_buttons(id="choice",label=ui.h6("Quelles sorties voulez-vous?"),choices=value_choice,inline=True),
                        ui.panel_conditional("input.choice ==='EigenRes'",
                            ui.br(),
                            ui.row(
                                ui.column(2,
                                    ui.input_radio_buttons(id="EigenChoice",label="Choice",choices={"eigenvalue" : "Eigenvalue","proportion" : "Proportion"},inline=False,selected="proportion"),
                                    ui.div(ui.input_switch(id="EigenLabel",label="Etiquettes",value=True),align="left")
                                ),
                                ui.column(10,ui.div(ui.output_plot("EigenPlot",width='100%',height='500px'),align="center"))
                            ),
                            ui.hr(),
                            PanelConditional1(text="",name="Eigen"),
                        ),
                        OverallPanelConditional(text="Mod"),
                        OverallPanelConditional(text="Ind"),
                        ui.panel_conditional(f"input.choice == 'VarRes'",
                            ui.br(),
                            ui.h5("Rapport de corrélation"),
                            PanelConditional1(text="Var",name="Eta2"),
                            ui.hr(),
                            ui.h5("Contributions"),
                            PanelConditional1(text="Var",name="Contrib")
                        ),
                        ui.output_ui("IndSupPanel"),
                        ui.output_ui("VarSupPanel"),  
                        ui.output_ui("VarQuantPanel")
                    ),
                    ui.nav_panel("Description automatique des axes",
                        ui.row(
                            ui.column(7,ui.input_radio_buttons(id="pvalueDimdesc",label="Probabilité critique",choices={x:y for x,y in zip([0.01,0.05,0.1,1.0],["Significance level 1%","Significance level 5%","Significance level 10%","None"])},selected=0.05,width="100%",inline=True)),
                            ui.column(5,ui.input_radio_buttons(id="Dimdesc",label="Choisir les dimensions",choices=DimDescChoice,selected="Dim.1",inline=True))
                        ),
                        ui.output_ui(id="DimDesc")
                    ),
                    ui.nav_panel("Résumé du jeu de données",
                        ui.input_radio_buttons(
                            id="ResumeChoice",
                            label=ui.h6("Quelles sorties voulez - vous?"),
                            choices={
                                "StatsDesc":"Statistiques descriptives",
                                "BarPlot" : "Bar plot"
                            },
                            selected="StatsDesc",
                            width="100%",
                            inline=True),
                            ui.br(),
                        ui.panel_conditional("input.ResumeChoice==='StatsDesc'",
                            ui.h5("Test du Chi2"),
                            PanelConditional1(text="Chi2",name="Test"),
                            ui.hr(),
                            ui.h5("Autres mesures d'association"),
                            PanelConditional1(text="Others",name="Test")
                        ),
                        ui.panel_conditional("input.ResumeChoice === 'BarPlot'",
                            ui.row(
                                ui.column(2,
                                    ui.input_select(id="VarLabel",label="Graphes sur",choices={x:x for x in VarLabelChoice},selected=VarLabelChoice[0]),
                                ),
                                ui.column(10,ui.div(ui.output_plot(id="VarBarPlotGraph",width='100%',height='500px'),align="center"))
                            )
                        )
                    ),
                    ui.nav_panel("Données",
                        PanelConditional1(text="OverallData",name="")
                    )
                )
            )
        )

        # Server
        def server(input:Inputs, output:Outputs, session:Session):
            
            #----------------------------------------------------------------------------------------------
            # Disable x and y axis
            @reactive.Effect
            def _():
                x = int(input.Axis1())
                Dim = [i for i in range(model.call_["n_components"]) if i > x]
                ui.update_select(
                    id="Axis2",
                    label="",
                    choices={x : x for x in Dim},
                    selected=Dim[0]
                )
            
            @reactive.Effect
            def _():
                x = int(input.Axis2())
                Dim = [i for i in range(model.call_["n_components"]) if i < x]
                ui.update_select(
                    id="Axis1",
                    label="",
                    choices={x : x for x in Dim},
                    selected=Dim[0]
                )
            
            #--------------------------------------------------------------------------------------------------
            @output
            @render.ui
            def IndTextChoice():
                if hasattr(model,"ind_sup_"):
                    return ui.TagList(
                        ui.input_select(
                            id="IndTextActifColor",
                            label="individus actifs",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="IndTextSupColor",
                            label="individus supplémentaires",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="blue",
                            multiple=False,
                            width="100%"
                        )
                    )
                else:
                    return ui.TagList(
                        ui.input_select(
                            id="IndTextActifColor",
                            label="individus actifs",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        )
                    )
            
            #-----------------------------------------------------------------------------------------
            # Disabled Individuals Text Colors
            @reactive.Effect
            def _():
                x = input.IndTextActifColor()
                Colors = [i for i in mcolors.CSS4_COLORS if i != x]
                ui.update_select(
                    id="IndTextSupColor",
                    label="individus supplémentaires",
                    choices={x : x for x in Colors},
                    selected="blue"
                )
            
            @reactive.Effect
            def _():
                x = input.IndTextSupColor()
                Dim = [i for i in mcolors.CSS4_COLORS if i != x]
                ui.update_select(
                    id="IndTextActifColor",
                    label="individus actifs",
                    choices={x : x for x in Dim},
                    selected="black"
                )
            
            #-------------------------------------------------------------------------------------------
            @output
            @render.ui
            def IndVarQuantColorPanel():
                if hasattr(model,"quanti_sup_"):
                    quanti_sup_labels = model.quanti_sup_["coord"].index.tolist()
                    return ui.TagList(
                        ui.input_select(
                            id="IndTextVarQuantColor",
                            label="Choix de la variable",
                            choices={x:x for x in quanti_sup_labels},
                            selected=quanti_sup_labels[0],
                            multiple=False
                        )
                    )
                else:
                    return ui.TagList(
                        ui.p(),
                        ui.p("Aucune variable quantitative")
                    )
                
            #-------------------------------------------------------------------------------------------
            @output
            @render.ui
            def ModTextChoice():
                if hasattr(model,"quali_sup_"):
                    return ui.TagList(
                        ui.input_select(
                            id="ModTextActifColor",
                            label="modalités actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="ModTextSupColor",
                            label="modalités supplémentaires",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="blue",
                            multiple=False,
                            width="100%"
                        ),
                    )
                else:
                    return ui.TagList(
                        ui.input_select(
                            id="ModTextActifColor",
                            label="modalités actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        )
                    )
            #-------------------------------------------------------------------------------------------
            # Disabled Varaibles Categories Text Colors
            @reactive.Effect
            def _():
                x = input.ModTextActifColor()
                Colors = [i for i in mcolors.CSS4_COLORS if i != x]
                ui.update_select(
                    id="ModTextSupColor",
                    label="modalités supplémentaires",
                    choices={x : x for x in Colors},
                    selected="blue"
                )
            
            @reactive.Effect
            def _():
                x = input.ModTextSupColor()
                Dim = [i for i in mcolors.CSS4_COLORS if i != x]
                ui.update_select(
                    id="ModTextActifColor",
                    label="modalités actives",
                    choices={x : x for x in Dim},
                    selected="black"
                )
            
            ##########################################################################################
            #
            #------------------------------------------------------------------------------------------
            @output
            @render.ui
            def VarPlotColorChoice():
                if model.quali_sup is not None and model.quanti_sup is not None:
                    return ui.TagList(
                        ui.input_select(
                            id="VarTextActifColor",
                            label="Variables qualitatives actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="VarSupTextSupColor",
                            label="Variables qualitatives supplémentaires",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="blue",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="VarQuantTextColor",
                            label="Variables quantitatives supplémentaires",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="red",
                            multiple=False,
                            width="100%"
                        )
                    )
                elif model.quali_sup is not None:
                    return ui.TagList(
                        ui.input_select(
                            id="VarTextActifColor",
                            label="Variables qualitatives actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="VarSupTextSupColor",
                            label="Variables qualitatives supplémentaires",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="blue",
                            multiple=False,
                            width="100%"
                        )
                    )
                elif model.quanti_sup is not None:
                    return ui.TagList(
                        ui.input_select(
                            id="VarTextActifColor",
                            label="Variables qualitatives actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="VarQuantTextColor",
                            label="Variables quantitatives supplémentaires",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="red",
                            multiple=False,
                            width="100%"
                        )
                    )
                else:
                    return ui.TagList(
                        ui.input_select(
                            id="VarTextActifColor",
                            label="Variables qualitatives actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        )
                    )
            
             #-------------------------------------------------------------------------------------------
            # Disabled Varaibles Categories Text Colors
            @reactive.Effect
            def _():
                x = input.VarTextActifColor()
                y = input.VarSupTextSupColor()
                Colors = [i for i in mcolors.CSS4_COLORS if i != x and i != y]
                ui.update_select(
                    id="VarQuantTextColor",
                    label="Variables quantitatives supplémentaires",
                    choices={x : x for x in Colors},
                    selected="red"
                )
            
            @reactive.Effect
            def _():
                x = input.VarTextActifColor()
                y = input.VarQuantTextColor()
                Colors = [i for i in mcolors.CSS4_COLORS if i != x and i != y]
                ui.update_select(
                    id="VarSupTextSupColor",
                    label="Variables qualitatives supplémentaires",
                    choices={x : x for x in Colors},
                    selected="blue"
                )

            @reactive.Effect
            def _():
                x = input.VarSupTextSupColor()
                y = input.VarQuantTextColor()
                Colors = [i for i in mcolors.CSS4_COLORS if i != x and i != y]
                ui.update_select(
                    id="VarTextActifColor",
                    label="Variables qualitatives actives",
                    choices={x : x for x in Colors},
                    selected="black"
                )

            #--------------------------------------------------------------------------------------------------------------
            @output
            @render.ui
            def VarQuantOutPut():
                if model.quanti_sup is not None:
                    return ui.TagList(
                        ui.div(ui.output_plot("VarQuantFactorMap",width='100%', height='500px'),align="center"),
                        ui.hr(),
                        ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                        ui.div(ui.download_button(id="VarQuantGraphDownloadJpg",label="jpg",style = download_btn_style,icon=None),style="display: inline-block;",align="center"),
                        ui.div(ui.download_button(id="VarQuantGraphDownloadPng",label="png",style = "background-color: #1C2951;"),style="display: inline-block;",align="center"),
                        ui.div(ui.download_button(id="VarQuantGraphDownloadPdf",label="pdf",style = "background-color: #1C2951;"),style="display: inline-block;",align="center"),
                    )
                
            @output
            @render.ui
            def VarQuantOptions():
                return ui.TagList(
                    ui.panel_conditional("input.IndModVar ==='VarQuantPlot'",
                        ui.input_text(
                                id="VarQuantTitle",
                                label='Titre du graphe',
                                value="Variables quantitatives supplémentaires",
                                width="100%"
                        ),
                        ui.input_slider(
                            id="VarQuantTextSize",
                            label="Taille des libellés",
                            min=8,
                            max=20,
                            value=8,
                            step=2,
                            ticks=False
                        )
                    )
                )
            
            #######################################################################################
            #   Supplementary Elements
            #---------------------------------------------------------------------------------------
            #-------------------------------------------------------------------------------------------------
            # Add individuals Supplementary Conditional Panel
            @output
            @render.ui
            def IndSupPanel():
                return ui.panel_conditional("input.choice == 'IndSupRes'",
                            ui.br(),
                            ui.h5("Coordonnées"),
                            PanelConditional1(text="IndSup",name="Coord"),
                            ui.hr(),
                            ui.h5("Cos2 - Qualité de la représentation"),
                            PanelConditional1(text="IndSup",name="Cos2") 
                        )
            
            # Add Variables/categories Supplementary Conditional Panel
            @output
            @render.ui
            def VarSupPanel():
                return ui.panel_conditional("input.choice == 'VarSupRes'",
                            ui.br(),
                            ui.h5("Coordonnées"),
                            PanelConditional1(text="VarSup",name="Coord"),
                            ui.hr(),
                            ui.h5("Cos2 - Qualité de la représentation"),
                            PanelConditional1(text="VarSup",name="Cos2"),
                            ui.hr(),
                            ui.h5("Vtest"),
                            PanelConditional1(text="VarSup",name="Vtest")
                        )
            
            # Add Continuous Supplementary Conditional Panel
            @output
            @render.ui
            def VarQuantPanel():
                return ui.panel_conditional("input.choice == 'VarQuantRes'",
                            ui.br(),
                            ui.h5("Coordonnées"),
                            PanelConditional1(text="VarQuant",name="Coord"),
                            ui.hr(),
                            ui.h5("Cos2 - Qualité de la représentation"),
                            PanelConditional1(text="VarQuant",name="Cos2"),
                        )
            
            #######################################################################################
            #   Tab : Graphes
            ######################################################################################

            ######################################################################################
            #   RowPLot
            #-----------------------------------------------------------------------------------------
            #--------------------------------------------------------------------------------
            @reactive.Calc
            def RowPlot():
                if input.IndTextColor() == "actif/sup":
                    if model.ind_sup is not None:
                        fig = fviz_mca_ind(self=model,
                                           axis=[int(input.Axis1()),int(input.Axis2())],
                                           color=input.IndTextActifColor(),
                                           color_sup = input.IndTextSupColor(),
                                           text_size = input.IndTextSize(),
                                           lim_contrib =input.IndLimContrib(),
                                           lim_cos2 = input.IndLimCos2(),
                                           title = input.IndTitle(),
                                           repel=input.IndPlotRepel(),
                                           ggtheme=pn.theme_gray())
                    else:
                        fig = fviz_mca_ind(self=model,
                                           axis=[int(input.Axis1()),int(input.Axis2())],
                                           color=input.IndTextActifColor(),
                                           text_size = input.IndTextSize(),
                                           lim_contrib =input.IndLimContrib(),
                                           lim_cos2 = input.IndLimCos2(),
                                           title = input.IndTitle(),
                                           repel=input.IndPlotRepel(),
                                           ggtheme=pn.theme_gray())
                elif input.IndTextColor() in ["cos2","contrib"]:
                    fig = fviz_mca_ind(self=model,
                                       axis=[int(input.Axis1()),int(input.Axis2())],
                                       color=input.IndTextColor(),
                                       text_size = input.IndTextSize(),
                                       lim_contrib =input.IndLimContrib(),
                                       lim_cos2 = input.IndLimCos2(),
                                       title = input.IndTitle(),
                                       repel=input.IndPlotRepel(),
                                       ggtheme=pn.theme_gray())
                elif input.IndTextColor() == "varqual":
                    fig = fviz_mca_ind(self=model,
                                       axis=[int(input.Axis1()),int(input.Axis2())],
                                       text_size = input.IndTextSize(),
                                       lim_contrib =input.IndLimContrib(),
                                       lim_cos2 = input.IndLimCos2(),
                                       title = input.IndTitle(),
                                       habillage= input.IndTextVarQualColor(),
                                       add_ellipses=input.IndAddEllipse(),
                                       repel=input.IndPlotRepel(),
                                       ggtheme=pn.theme_gray())
                elif  input.IndTextColor() == "varquant":
                    if model.quanti_sup is not None:
                        fig = fviz_mca_ind(self=model,
                                           axis=[int(input.Axis1()),int(input.Axis2())],
                                           color=input.IndTextVarQuantColor(),
                                           text_size = input.IndTextSize(),
                                           lim_contrib =input.IndLimContrib(),
                                           lim_cos2 = input.IndLimCos2(),
                                           title = input.IndTitle(),
                                           habillage= None,
                                           add_ellipses=input.IndAddEllipse(),
                                           repel=input.IndPlotRepel(),
                                           ggtheme=pn.theme_gray())
                    else:
                        fig = pn.ggplot()

                return fig

            # ------------------------------------------------------------------------------
            # Individual Factor Map - PCA
            @output
            @render.plot(alt="Individuals Factor Map - MCA")
            def RowFactorMap():
                return RowPlot().draw()
            
            import io
            # @session.download(filename="Individuals-Factor-Map.png")
            # def IndGraphDownloadPng():
            #     with io.BytesIO() as buf:
            #         plt.savefig(RowPlot(), format="png")
            #         yield buf.getvalue()

            #####################################################################################
            #   Variables Categories Plot
            #------------------------------------------------------------------------------------
             #############################################################################################
            #  Variables Factor Map
            ##############################################################################################

            @reactive.Calc
            def ModFactorPlot():
                if input.ModTextColor() == "actif/sup":
                    if model.quali_sup is not None:
                        fig = fviz_mca_mod(self=model,
                                           axis=[int(input.Axis1()),int(input.Axis2())],
                                           title=input.ModTitle(),
                                           color=input.ModTextActifColor(),
                                           color_sup=input.ModTextSupColor(),
                                           text_size=input.ModTextSize(),
                                           lim_contrib = input.ModLimContrib(),
                                           lim_cos2 = input.ModLimCos2(),
                                           repel=input.ModPlotRepel(),
                                           ggtheme=pn.theme_gray())
                    else:
                        fig = fviz_mca_mod(self=model,
                                           axis=[int(input.Axis1()),int(input.Axis2())],
                                           title=input.ModTitle(),
                                           color=input.ModTextActifColor(),
                                           color_sup=None,
                                           text_size=input.ModTextSize(),
                                           lim_contrib = input.ModLimContrib(),
                                           lim_cos2 = input.ModLimCos2(),
                                           repel=input.ModPlotRepel(),
                                           ggtheme=pn.theme_gray())
                elif input.ModTextColor() in ["cos2","contrib"]:
                    if model.quali_sup is not None:
                        fig = fviz_mca_mod(self=model,
                                           axis=[int(input.Axis1()),int(input.Axis2())],
                                           title=input.ModTitle(),
                                           color=input.ModTextColor(),
                                           color_sup=input.ModTextSupColor(),
                                           text_size=input.ModTextSize(),
                                           lim_contrib = input.ModLimContrib(),
                                           lim_cos2 = input.ModLimCos2(),
                                           repel=input.ModPlotRepel(),
                                           ggtheme=pn.theme_gray())
                    else:
                        fig = fviz_mca_mod(self=model,
                                           axis=[int(input.Axis1()),int(input.Axis2())],
                                           title=input.ModTitle(),
                                           color=input.ModTextColor(),
                                           color_sup=None,
                                           text_size=input.ModTextSize(),
                                           lim_contrib = input.ModLimContrib(),
                                           lim_cos2 = input.ModLimCos2(),
                                           repel=input.ModPlotRepel(),
                                           ggtheme=pn.theme_gray())
                return fig

            # Variables categories Factor Map - MCA
            @output
            @render.plot(alt="Variables categories Factor Map - MCA")
            def ModFactorMap():
                return ModFactorPlot().draw()
            
            #################################################################################################
            # Variables Map
            #-------------------------------------------------------------------------------------------------
            @reactive.Calc
            def VarFactorPlot():
                if (model.quali_sup is not None) and (model.quanti_sup is not None):
                    fig = fviz_mca_var(self=model,
                                       axis=[int(input.Axis1()),int(input.Axis2())],
                                       title=input.VarTitle(),
                                       color=input.VarTextActifColor(),
                                       color_sup=input.VarSupTextSupColor(),
                                       color_quanti_sup=input.VarQuantTextColor(),
                                       text_size=input.VarTextSize(),
                                       repel=input.VarPlotRepel(),
                                       ggtheme=pn.theme_gray())
                elif model.quali_sup is not None:
                    fig = fviz_mca_var(self=model,
                                       axis=[int(input.Axis1()),int(input.Axis2())],
                                       title=input.VarTitle(),
                                       color=input.VarTextActifColor(),
                                       color_sup=input.VarSupTextSupColor(),
                                       text_size=input.VarTextSize(),
                                       repel=input.VarPlotRepel(),
                                       ggtheme=pn.theme_gray())
                elif model.quanti_sup is not None:
                    fig = fviz_mca_var(self=model,
                                       axis=[int(input.Axis1()),int(input.Axis2())],
                                       title=input.VarTitle(),
                                       color=input.VarTextActifColor(),
                                       color_quanti_sup=input.VarQuantTextColor(),
                                       text_size=input.VarTextSize(),
                                       repel=input.VarPlotRepel(),
                                       ggtheme=pn.theme_gray())
                else:
                    fig = fviz_mca_var(self=model,
                                       axis=[int(input.Axis1()),int(input.Axis2())],
                                       title=input.VarTitle(),
                                       color=input.VarTextActifColor(),
                                       text_size=input.VarTextSize(),
                                       repel=input.VarPlotRepel(),
                                       ggtheme=pn.theme_gray())
                return fig

            # Variables Factor Map - MCA
            @output
            @render.plot(alt="Variables Factor Map - MCA")
            def VarFactorMap():
                return VarFactorPlot().draw()
            
            #################################################################################################
            #   Supplementary Continuous variables
            #-------------------------------------------------------------------------------------------------
            # Supplementary Continuous variables MAP
            @reactive.Calc
            def VarQuantFactorPlot():
                fig =  fviz_corrcircle(self=model,
                                       axis=[int(input.Axis1()),int(input.Axis2())],
                                       title=input.VarQuantTitle(),
                                       text_size=input.VarQuantTextSize(),
                                       ggtheme=pn.theme_gray())
                return fig
            
            @output
            @render.plot(alt="Variables quantitatives supplémentaires Factor Map - MCA")
            def VarQuantFactorMap():
                return VarQuantFactorPlot().draw()

            ##################################################################################################
            #   Tab Valeurs
            ##################################################################################################
            #-------------------------------------------------------------------------------------------
            # Eigenvalue - Scree plot
            @output
            @render.plot(alt="Scree Plot - MCA")
            def EigenPlot():
                EigenFig = fviz_eig(self=model,
                                    choice=input.EigenChoice(),
                                    add_labels=input.EigenLabel(),
                                    ggtheme=pn.theme_gray())
                return EigenFig.draw()
            
            # Eigen value - DataFrame
            @render.data_frame
            def EigenTable():
                EigenData = model.eig_.round(4).reset_index().rename(columns={"index":"dimensions"})
                EigenData.columns = [x.capitalize() for x in EigenData.columns]
                return DataTable(data=match_datalength(EigenData,input.EigenLen()),filters=input.EigenFilter())
            
            #---------------------------------------------------------------------------------------------
            #################################################################################################
            #   Categories/modalités
            #####################################################################################################
            #---------------------------------------------------------------------------------------------
            # Variables Coordinates
            @output
            @render.data_frame
            def ModCoordTable():
                ModCoord = model.var_["coord"].round(4).reset_index()
                ModCoord.columns = ["Categories", *ModCoord.columns[1:]]
                return DataTable(data=match_datalength(data=ModCoord,value=input.ModCoordLen()),filters=input.ModCoordFilter())
            
            #----------------------------------------------------------------------------------------------------
            # Variables Contributions
            @output
            @render.data_frame
            def ModContribTable():
                ModContrib = model.var_["contrib"].round(4).reset_index()
                ModContrib.columns = ["Categories", *ModContrib.columns[1:]]
                return  DataTable(data=match_datalength(data=ModContrib,value=input.ModContribLen()),filters=input.ModContribFilter())
            
            #-----------------------------------------------------------------------------------------------------
            # Add Variables Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.ModContribGraphBtn)
            def _():
                GraphModalShow(text="Mod",name="Contrib",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def ModContribMap():
                fig = fviz_contrib(self=model,
                                   choice="var",
                                   axis=input.ModContribAxis(),
                                   top_contrib=int(input.ModContribTop()),
                                   color=input.ModContribColor(),
                                   bar_width=input.ModContribBarWidth(),
                                   ggtheme=pn.theme_gray())
                return fig

            # Plot variables Contributions
            @output
            @render.plot(alt="Variables categories contributions Map - MCA")
            def ModContribPlot():
                return ModContribMap().draw()
            
            #-----------------------------------------------------------------------------------------------------------
            # Variables categories Cos2 
            @output
            @render.data_frame
            def ModCos2Table():
                ModCos2 = model.var_["cos2"].round(4).reset_index()
                ModCos2.columns = ["Categories", *ModCos2.columns[1:]]
                return  DataTable(data=match_datalength(data=ModCos2,value=input.ModCos2Len()),filters=input.ModCos2Filter())
            
            #-------------------------------------------------------------------------------------------------------------
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.ModCos2GraphBtn)
            def _():
                GraphModalShow(text="Mod",name="Cos2",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def ModCos2Map():
                fig = fviz_cos2(self=model,
                                choice = "var",
                                axis=input.ModCos2Axis(),
                                top_cos2=int(input.ModCos2Top()),
                                color=input.ModCos2Color(),
                                bar_width=input.ModCos2BarWidth(),
                                ggtheme=pn.theme_gray())
                return fig

            # Plot variables categories Cos2
            @output
            @render.plot(alt="Variables categories Cosines Map - MCA")
            def ModCos2Plot():
                return ModCos2Map().draw()
            
            ########################################################################################################
            # Individuals informations
            #---------------------------------------------------------------------------------------------
            # Individuals Coordinates
            @output
            @render.data_frame
            def IndCoordTable():
                IndCoord = model.ind_["coord"].round(4).reset_index()
                IndCoord.columns = ["Individus", *IndCoord.columns[1:]]
                return DataTable(data = match_datalength(IndCoord,input.IndCoordLen()),filters=input.IndCoordFilter())
            
            # Individuals Contributions
            @output
            @render.data_frame
            def IndContribTable():
                IndContrib = model.ind_["contrib"].round(4).reset_index()
                IndContrib.columns = ["Individus", *IndContrib.columns[1:]]
                return  DataTable(data=match_datalength(IndContrib,input.IndContribLen()),filters=input.IndContribFilter())
            
            # Add indiviuals Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.IndContribGraphBtn)
            def _():
                GraphModalShow(text="Ind",name="Contrib",max_axis=model.call_["n_components"])

            # Plot Individuals Contributions
            @output
            @render.plot(alt="Individuals Contributions Map - MCA")
            def IndContribPlot():
                IndContribFig = fviz_contrib(self=model,
                                             choice="ind",
                                             axis=input.IndContribAxis(),
                                             top_contrib=int(input.IndContribTop()),
                                             color = input.IndContribColor(),
                                             bar_width= input.IndContribBarWidth(),
                                             ggtheme=pn.theme_gray())
                return IndContribFig.draw()
            
            # Individuals Cos2 
            @output
            @render.data_frame
            def IndCos2Table():
                IndCos2 = model.ind_["cos2"].round(4).reset_index()
                IndCos2.columns = ["Individus", *IndCos2.columns[1:]]
                return  DataTable(data = match_datalength(IndCos2,input.IndCos2Len()),filters=input.IndCos2Filter())
            
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.IndCos2GraphBtn)
            def _():
                GraphModalShow(text="Ind",name="Cos2",max_axis=model.call_["n_components"])

            # Plot variables Cos2
            @output
            @render.plot(alt="Individuals Cosines Map - MCA")
            def IndCos2Plot():
                IndCos2Fig = fviz_cos2(self=model,
                                       choice="ind",
                                       axis=input.IndCos2Axis(),
                                       top_cos2=int(input.IndCos2Top()),
                                       color=input.IndCos2Color(),
                                       bar_width=input.IndCos2BarWidth(),
                                       ggtheme=pn.theme_gray())
                return IndCos2Fig.draw()
            
            ###########################################################################################
            #   Variables informations
            #------------------------------------------------------------------------------------------
            # Variables - Rapport de corrélation
            @render.data_frame
            def VarEta2Table():
                VarEta2 = model.var_["eta2"].round(4).reset_index()
                VarEta2.columns = ["Variables", *VarEta2.columns[1:]]
                return DataTable(data = match_datalength(VarEta2,input.VarEta2Len()),filters=input.VarEta2Filter())
            
            # Variables Contributions
            @render.data_frame
            def VarContribTable():
                VarContrib = model.var_["var_contrib"].round(4).reset_index()
                VarContrib.columns = ["Variables", *VarContrib.columns[1:]]
                return  DataTable(data=match_datalength(VarContrib,input.VarContribLen()),filters=input.VarContribFilter())
            
            ############################################################################################
            #    Supplementary individuals informations
            #-------------------------------------------------------------------------------------------
            # Supplementary Individual Coordinates
            @output
            @render.data_frame
            def IndSupCoordTable():
                IndSupCoord = model.ind_sup_["coord"].round(4).reset_index()
                IndSupCoord.columns = ["Individus", *IndSupCoord.columns[1:]]
                return  DataTable(data = match_datalength(IndSupCoord,input.IndSupCoordLen()),filters=input.IndSupCoordFilter())
            
            # Supplementaru Individual Cos2
            @output
            @render.data_frame
            def IndSupCos2Table():
                IndSupCos2 = model.ind_sup_["cos2"].round(4).reset_index()
                IndSupCos2.columns = ["Individus", *IndSupCos2.columns[1:]]
                return  DataTable(data = match_datalength(IndSupCos2,input.IndSupCos2Len()),filters=input.IndSupCos2Filter())
            
            ##########################################################################################
            # Supplementary continuous variables
            #-----------------------------------------------------------------------------------------
            # Supplementary continuous variables coordinates
            @output
            @render.data_frame
            def VarQuantCoordTable():
                VarQuantCoord = model.quanti_sup_["coord"].round(4).reset_index()
                VarQuantCoord.columns = ["Individus", *VarQuantCoord.columns[1:]]
                return  DataTable(data = match_datalength(VarQuantCoord,input.VarQuantCoordLen()),filters=input.VarQuantCoordFilter())
            
            # Supplementary continuous variables cos2
            @output
            @render.data_frame
            def VarQuantCos2Table():
                VarQuantCos2 = model.quanti_sup_["cos2"].round(4).reset_index()
                VarQuantCos2.columns = ["Individus", *VarQuantCos2.columns[1:]]
                return  DataTable(data = match_datalength(VarQuantCos2,input.VarQuantCos2Len()),filters=input.VarQuantCos2Filter())

            ###########################################################################################
            # Supplementary variables categories
            #-------------------------------------------------------------------------------------------
            ## Supplementary Variables/categories coordinates
            @render.data_frame
            def VarSupCoordTable():
                VarSupCoord = model.quali_sup_["coord"].round(4).reset_index()
                VarSupCoord.columns = ["Categories", *VarSupCoord.columns[1:]]
                return DataTable(data=match_datalength(data=VarSupCoord,value=input.VarSupCoordLen()),filters=input.VarSupCoordFilter())
            
            # Supplementary variables/categories Cos2
            @render.data_frame
            def VarSupCos2Table():
                VarSupCos2 = model.quali_sup_["cos2"].round(4).reset_index()
                VarSupCos2.columns = ["Categories", *VarSupCos2.columns[1:]]
                return DataTable(data=match_datalength(data=VarSupCos2,value=input.VarSupCos2Len()),filters=input.VarSupCos2Filter())
            
            # Supplementary variables/categories Vtest
            @render.data_frame
            def VarSupVtestTable():
                VarSupVtest = model.quali_sup_["vtest"].round(4).reset_index()
                VarSupVtest.columns = ["Categories", *VarSupVtest.columns[1:]]
                return DataTable(data=match_datalength(data=VarSupVtest,value=input.VarSupVtestLen()),filters=input.VarSupVtestFilter())
            
            ############################################################################################
            #    Tab : Description automatique des axes
            ############################################################################################
            #---------------------------------------------------------------------------------------
            # Description of axis
            @output
            @render.ui
            def DimDesc():
                if model.quanti_sup is not None:
                    return ui.TagList(
                        ui.h5("Variables qualitative"),
                        PanelConditional1(text="Dim1",name="Desc"),
                        ui.hr(),
                        ui.h5("Variables quantitatives"),
                        PanelConditional1(text="Dim2",name="Desc"),
                    )
                else:
                    return ui.TagList(
                        ui.h5("Variables qualitatives"),
                        PanelConditional1(text="Dim1",name="Desc")
                    )
            
            #--------------------------------------------------------------------------------------------------------
            @output
            @render.data_frame
            def Dim1DescTable():
                DimDesc = dimdesc(self=model,axis=None,proba=float(input.pvalueDimdesc()))
                if isinstance(DimDesc[input.Dimdesc()],dict):
                    DimDescQuali = DimDesc[input.Dimdesc()]["quali"].reset_index().rename(columns={"index":"Variables"})
                elif isinstance(DimDesc[input.Dimdesc()],pd.DataFrame):
                    DimDescQuali = DimDesc[input.Dimdesc()].reset_index().rename(columns={"index":"Variables"})
                else:
                    DimDescQuali = pd.DataFrame()
                return  DataTable(data = match_datalength(DimDescQuali,input.Dim1DescLen()),
                                filters=input.Dim1DescFilter())
            
            #--------------------------------------------------------------------------------------------------
            @output
            @render.data_frame
            def Dim2DescTable():
                DimDesc = dimdesc(self=model,axis=None,proba=float(input.pvalueDimdesc()))
                if isinstance(DimDesc[input.Dimdesc()],dict):
                    DimDescQuanti = DimDesc[input.Dimdesc()]["quanti"].reset_index().rename(columns={"index":"Variables"})
                else:
                    DimDescQuanti = pd.DataFrame()
                return  DataTable(data = match_datalength(DimDescQuanti,input.Dim2DescLen()),
                                  filters=input.Dim2DescFilter())
            
            ###########################################################################################
            #   Tab : Resumé du jeu de données
            ###########################################################################################
            #-----------------------------------------------------------------------------------------------
            ### Chi2 statistic test
            @render.data_frame
            def Chi2TestTable():
                return  DataTable(data = match_datalength(model.chi2_test_,input.Chi2TestLen()),filters=input.Chi2TestFilter())
            
            @render.data_frame
            def OthersTestTable():
                # Create Data
                data = model.call_["X"]
                if model.quali_sup is not None:
                    X_quali_sup = model.call_["Xtot"].loc[:,model.quali_sup_["eta2"].index.tolist()].astype("object")
                    if model.ind_sup is not None:
                        X_quali_sup = X_quali_sup.drop(index=[name for name in model.call_["Xtot"].index.tolist() if name in model.ind_sup_["coord"].index.tolist()])
                    data = pd.concat([data,X_quali_sup],axis=1)

                others_test = pd.DataFrame(columns=["variable1","variable2","cramer","tschuprow","pearson"]).astype("float")
                idx = 0
                for i in np.arange(data.shape[1]-1):
                    for j in np.arange(i+1,data.shape[1]):
                        tab = pd.crosstab(data.iloc[:,i],data.iloc[:,j])
                        row_others = pd.DataFrame({"variable1" : data.columns.tolist()[i],
                                                   "variable2" : data.columns.tolist()[j],
                                                   "cramer"    : sp.stats.contingency.association(tab,method="cramer"),
                                                   "tschuprow" : sp.stats.contingency.association(tab,method="tschuprow"),
                                                   "pearson"   : sp.stats.contingency.association(tab,method="pearson")},
                                                   index=[idx])
                        others_test = pd.concat((others_test,row_others),axis=0,ignore_index=True)
                        idx = idx + 1
                return  DataTable(data = match_datalength(others_test,input.OthersTestLen()),filters=input.OthersTestFilter())

            #---------------------------------------------------------------------------------------------
            # Diagramme en barres
            @output
            @render.plot(alt="Bar-Plot")
            def VarBarPlotGraph():
                data = model.call_["X"]
                if model.quali_sup is not None:
                    X_quali_sup = model.call_["Xtot"].loc[:,model.quali_sup_["eta2"].index.tolist()].astype("object")
                    if model.ind_sup is not None:
                        X_quali_sup = X_quali_sup.drop(index=[name for name in model.call_["Xtot"].index.tolist() if name in model.ind_sup_["coord"].index.tolist()])
                    data = pd.concat([data,X_quali_sup],axis=1)
                p = pn.ggplot(data,pn.aes(x=input.VarLabel()))+ pn.geom_bar()
                return p.draw()
            
            ##############################################################################################
            # Tab : Data
            ###############################################################################################################
            #-------------------------------------------------------------------------------------------------
            # Overall Data
            @output
            @render.data_frame
            def OverallDataTable():
                overalldata = model.call_["Xtot"].reset_index()
                return DataTable(data = match_datalength(overalldata,input.OverallDataLen()),filters=input.OverallDataFilter())
            
            ###########################################################
            # Exit
            # -------------------------------------------------------
            @reactive.Effect
            @reactive.event(input.exit)
            async def _():
                await session.close()
            
        self.app_ui = app_ui
        self.app_server = server
    
    def run(self,**kwargs):

        """
        Run the app

        Parameters:
        ----------
        kwargs : objet = {}. See https://shiny.posit.co/py/api/App.html
        
        """

        app = App(ui=self.app_ui, server=self.app_server)
        return app.run(**kwargs)
    
    # Run with notebooks
    def run_notebooks(self,**kwargs):

        nest_asyncio.apply()
        uvicorn.run(self.run(**kwargs))
    
    def stop(self):
        """
        
        
        """
        app = App(ui=self.app_ui, server=self.server)
        return app.stop()


