# -*- coding: utf-8 -*-
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
import shinyswatch
from pathlib import Path
import pandas as pd
import plotnine as pn
import matplotlib.colors as mcolors
import nest_asyncio
import uvicorn

#  All scientisttools fonctions
from scientisttools import fviz_famd_ind, fviz_famd_mod,fviz_famd_var,fviz_famd_col,fviz_eig, fviz_contrib,fviz_cos2,fviz_corrplot, fviz_corrcircle,dimdesc

from .function import *

colors = mcolors.CSS4_COLORS
colors["cos2"] = "cos2"
colors["contrib"] = "contrib"

css_path = Path(__file__).parent / "www" / "style.css"

class FAMDshiny:
    """
    Factor Analysis of Mixed Data (FAMD) with scientistshiny
    --------------------------------------------------------

    Description
    -----------
    Performs Factor Analysis of Mixed Data (FAMD) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables on a Shiny application.
    Graphics can be downloaded in png, jpg and pdf.

    Usage
    -----
    FAMDshiny(model)

    Parameters:
    ----------
    fa_model : An instance of class FAMD. A FAMD result from scientisttools.

    Returns:
    -------
    Graphs : a tab containing the individuals factor map and the variables factor map

    Values : a tab containing the summary of the FAMD performed, the eigenvalue, the results
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
    Duvérier DJIFACK ZEBAZE : duverierdjifack@gmail.com

    Examples:
    ---------
    > from scientisttools import FAMD
    > from scientistshiny import FAMDshiny


    for jupyter notebooks
    https://stackoverflow.com/questions/74070505/how-to-run-fastapi-application-inside-jupyter
    """


    def __init__(self,model=None):
        # Check if model FAMD
        if model.model_ != "famd":
            raise ValueError("'model' must be an instance of class FAMD")
        
        # -----------------------------------------------------------------------------------
        # Initialise value choice
        value_choice = {"EigenRes"      : "Valeurs propres",
                        "ModRes"        : "Résultats des modalités",
                        "IndRes"        : "Résultats sur les individus",
                        "VarQuantRes"   : "Résultats sur les variables quantitatives",
                        "VarQualRes"    : "Résultats sur les variables qualitatives"}
        
        if model.ind_sup is not None:
            value_choice = {**value_choice,**{"IndSupRes" : "Résultats des individus supplémentaires"}}
            
        # Check if supplementary quantitatives variables
        if model.quanti_sup is not None:
            value_choice = {**value_choice, **{"VarQuantSupRes" : "Résultats des variables quantitatives supplémentaires"}}
        
        # Check if supplementary qualitatives variables
        if model.quali_sup is not None:
            value_choice = {**value_choice,**{"ModSupRes"     : "Résultats des modalités supplémentaires",
                                              "VarQualSupRes" : "Résultats des variables qualitatives supplémentaires"}}
        
        # Dim Desc Choice
        DimDescChoice = {}
        for i in range(min(3,model.call_["n_components"])):
            DimDescChoice.update({"Dim."+str(i+1) : "Dimension "+str(i+1)})

        # Quantitatives variables (Actives + supplementary)
        VarQuantLabelChoice = model.quanti_var_["coord"].index.tolist()
        if model.quanti_sup is not None:
            VarQuantLabelChoice = [*VarQuantLabelChoice,*model.quanti_sup_["coord"].index.tolist()]
         
        # Qualitatives variables (Actives + Supplementary)
        VarQualLabelChoice = [x for x in model.var_["coord"].index.tolist() if x  not in VarQuantLabelChoice]
        if model.quali_sup is not None:
            VarQualLabelChoice = [*VarQualLabelChoice,*model.quali_sup_["eta2"].index.tolist()]
        
        # UI
        app_ui = ui.page_fluid(
            ui.include_css(css_path),
            shinyswatch.theme.superhero(),
            ui.page_navbar(
                title=ui.div(ui.panel_title(ui.h2("Analyse Factorielle des Données Mixtes"),window_title="FAMDshiny"),
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
                    ui.input_select(
                        id="IndModVar",
                        label="Quel graphe voulez-vous modifier?",
                        choices={
                            "IndPlot"      : "Individus",
                            "ModPlot"      : "Modalités",
                            "VarPlot"      : "Variables",
                            "VarQuantPlot" : "Variables quantitatives"
                        },
                        selected="IndPlot",
                        width="100%"
                    ),
                    ui.panel_conditional("input.IndModVar ==='IndPlot'",
                        ui.input_text(
                            id="IndTitle",
                            label="Titre du graphe",
                            value="Individuals Factor Map - FAMD",
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
                                 choices={x:x for x in VarQualLabelChoice},
                                 selected=VarQualLabelChoice[0],
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
                            ui.input_select(
                            id="IndTextVarQuantColor",
                            label="Choix de la variable",
                            choices={x:x for x in VarQuantLabelChoice},
                            selected=VarQuantLabelChoice[0],
                            multiple=False
                        )
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
                            value="Qualitative variable categories - FAMD",
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
                                value="Graphe des variables - FAMD",
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
                        ui.input_select(
                            id="VarPointSelect",
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
                        ui.panel_conditional("input.VarPointSelect === 'cos2'",
                            ui.div(ui.input_slider(id="VarLimCos2", label = "Libellés pour un cos2 plus grand que",min = 0, max = 1,value=0,step=0.05),align="center")              
                        ),
                         ui.panel_conditional("input.VarPointSelect === 'contrib'",
                            ui.div(ui.input_slider(id="VarLimContrib", label = "Libellés pour une contribution plus grande que",min = 0, max = 100,value=0,step=5),align="center")              
                        ),
                        ui.output_ui("VarPlotColorChoice"),
                        ui.input_switch(
                            id="VarPlotRepel",
                            label="repel",
                            value=True
                        )
                    ),
                    ui.panel_conditional("input.IndModVar ==='VarQuantPlot'",
                        ui.input_text(
                            id="VarQuantTitle",
                            label='Titre du graphe',
                            value="Cercle des corrélations",
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
                        ),
                        ui.input_select(
                            id="VarQuantPointSelect",
                            label="Représenter les variables en fonction de:",
                            choices={
                                "none"    : "Pas de sélection",
                                "cos2"    : "Cosinus",
                                "contrib" : "Contribution"
                                },
                            selected="none",
                            multiple=False,
                            width="100%"
                        ),
                        ui.panel_conditional("input.VarQuantPointSelect === 'cos2'",
                            ui.div(ui.input_slider(id="VarQuantLimCos2", label = "Libellés pour un cos2 plus grand que",min = 0, max = 1,value=0,step=0.05),align="center")              
                        ),
                        ui.panel_conditional("input.VarQuantPointSelect === 'contrib'",
                            ui.div(ui.input_slider(id="VarQuantLimContrib", label ="Libellés pour une contribution plus grande que",min = 0, max = 100,value=0,step=5),align="center")              
                        ),
                        ui.input_select(
                            id="VarQuantTextColor",
                            label="Colorier les points par :",
                            choices={
                                "actif/sup": "actifs/supplémentaires",
                                "cos2"     : "Cosinus",
                                "contrib"  : "Contribution"
                            },
                            selected="actif/sup",
                            multiple=False,
                            width="100%"
                        )
                    ),
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
                                ui.div(ui.output_plot("VarQuantFactorMap",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                                ui.div(ui.download_button(id="VarQuantGraphDownloadJpg",label="jpg",style = download_btn_style,icon=None),style="display: inline-block;",align="center"),
                                ui.div(ui.download_button(id="VarQuantGraphDownloadPng",label="png",style = "background-color: #1C2951;"),style="display: inline-block;",align="center"),
                                ui.div(ui.download_button(id="VarQuantGraphDownloadPdf",label="pdf",style = "background-color: #1C2951;"),style="display: inline-block;",align="center"),
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
                        OverallPanelConditional(text="VarQuant"),
                        ui.panel_conditional(f"input.choice == 'VarQualRes'",
                            ui.br(),
                            ui.h5("Rapport de corrélation"),
                            PanelConditional1(text="VarQual",name="Eta2"),
                            ui.hr(),
                            ui.h5("Contributions"),
                            PanelConditional1(text="VarQual",name="Contrib"),
                            ui.hr(),
                            ui.h5("Cos2 - Qualité de la représentation"),
                            PanelConditional1(text="VarQual",name="Cos2")
                        ),
                        ui.output_ui("IndSupPanel"),
                        ui.output_ui("VarQuantSupPanel"),
                        ui.output_ui("ModSupPanel"),
                        ui.output_ui("VarQualSupPanel")
                    ),
                    ui.nav_panel("Description automatique des axes",
                        ui.row(
                            ui.column(7,ui.input_radio_buttons(id="pvalueDimdesc",label="Probabilité critique",choices={x:y for x,y in zip([0.01,0.05,0.1,1.0],["Significance level 1%","Significance level 5%","Significance level 10%","None"])},selected=0.05,width="100%",inline=True)),
                            ui.column(5,ui.input_radio_buttons(id="Dimdesc",label="Choisir les dimensions",choices=DimDescChoice,selected="Dim.1",inline=True))
                        ),
                        ui.h5("Variables quantitatives"),
                        PanelConditional1(text="Dim1",name="Desc"),
                        ui.hr(),
                        ui.h5("Variables qualitatives"),
                        PanelConditional1(text="Dim2",name="Desc")
                    ),
                    ui.nav_panel("Résumé du jeu de données",
                        ui.input_radio_buttons(
                            id="ResumeChoice",
                            label=ui.h6("Quelles sorties voulez - vous?"),
                            choices={
                                "StatsDesc":"Statistiques descriptives",
                                "Hist" : "Histogramme",
                                "CorrMatrix": "Matrice des corrélations",
                                "BarPlot" : "Bar plot"
                            },
                            selected="StatsDesc",
                            width="100%",
                            inline=True),
                        ui.panel_conditional("input.ResumeChoice==='StatsDesc'",
                            PanelConditional1(text="StatsDesc",name="")
                        ),
                        ui.panel_conditional("input.ResumeChoice === 'Hist'",
                            ui.row(
                                ui.column(2,
                                    ui.input_select(
                                        id="VarQuantLabel",
                                        label="Choisir une variable",
                                        choices={x:x for x in VarQuantLabelChoice},
                                        selected=VarQuantLabelChoice[0]
                                        ),
                                    ui.input_switch(
                                        id="AddDensity",
                                        label="Densite",
                                        value=False
                                    )

                                ),
                                ui.column(10,
                                    ui.div(
                                        ui.output_plot(
                                            id="VarHistGraph",
                                            width='100%',
                                            height='500px'
                                        ),
                                        align="center"
                                    )
                                )
                            )
                        ),
                        ui.panel_conditional("input.ResumeChoice==='CorrMatrix'",
                            PanelConditional1(text="CorrMatrix",name="")
                        ),
                        ui.panel_conditional("input.ResumeChoice === 'BarPlot'",
                            ui.row(
                                ui.column(2,
                                    ui.input_select(id="VarQualLabel",label="Graphes sur",choices={x:x for x in VarQualLabelChoice},selected=VarQualLabelChoice[0]),
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
                if model.ind_sup is not None:
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
            def ModTextChoice():
                if model.quali_sup is not None:
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
                            id="VarQualActifTextColor",
                            label="Variables qualitatives actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="VarQuantActifTextColor",
                            label="Variables quantitatives actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="blue",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="VarQualSupTextColor",
                            label="Variables qualitatives supplémentaires",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="red",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="VarQuantSupTextColor",
                            label="Variables quantitatives supplémentaires",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="pink",
                            multiple=False,
                            width="100%"
                        )
                    )
                elif model.quali_sup is not None:
                    return ui.TagList(
                        ui.input_select(
                            id="VarQualActifTextColor",
                            label="Variables qualitatives actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="VarQuantActifTextColor",
                            label="Variables quantitatives actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="blue",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="VarQualSupTextColor",
                            label="Variables qualitatives supplémentaires",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="red",
                            multiple=False,
                            width="100%"
                        )
                    )
                elif model.quanti_sup is not None:
                    return ui.TagList(
                        ui.input_select(
                            id="VarQualActifTextColor",
                            label="Variables qualitatives actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="VarQuantActifTextColor",
                            label="Variables quantitatives actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="blue",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="VarQuantSupTextColor",
                            label="Variables quantitatives supplémentaires",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="pink",
                            multiple=False,
                            width="100%"
                        )
                    )
                else:
                    return ui.TagList(
                        ui.input_select(
                            id="VarQualActifTextColor",
                            label="Variables qualitatives actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="VarQuantActifTextColor",
                            label="Variables quantitatives actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="blue",
                            multiple=False,
                            width="100%"
                        )
                    )
            
             #-------------------------------------------------------------------------------------------
            # Disabled Varaibles Categories Text Colors
            @reactive.Effect
            def _():
                x = input.VarQualActifTextColor()
                y = input.VarQuantActifTextColor()
                z = input.VarQualSupTextColor()
                Colors = [i for i in mcolors.CSS4_COLORS if i not in [x,y,z]]
                ui.update_select(
                    id="VarQuantSupTextColor",
                    label="Variables quantitatives supplémentaires",
                    choices={x : x for x in Colors},
                    selected="pink"
                )
            
            @reactive.Effect
            def _():
                x = input.VarQualActifTextColor()
                y = input.VarQuantActifTextColor()
                z = input.VarQuantSupTextColor()
                Colors = [i for i in mcolors.CSS4_COLORS if i not in [x,y,z]]
                ui.update_select(
                    id="VarQualSupTextColor",
                    label="Variables qualitatives supplémentaires",
                    choices={x : x for x in Colors},
                    selected="red"
                )
            
            @reactive.Effect
            def _():
                x = input.VarQualActifTextColor()
                y = input.VarQualSupTextColor()
                z = input.VarQuantSupTextColor()
                Colors = [i for i in mcolors.CSS4_COLORS if i not in [x,y,z]]
                ui.update_select(
                    id="VarQuantActifTextColor",
                    label="Variables quantitatives",
                    choices={x : x for x in Colors},
                    selected="blue"
                )
            
            @reactive.Effect
            def _():
                x = input.VarQuantActifTextColor()
                y = input.VarQualSupTextColor()
                z = input.VarQuantSupTextColor()
                Colors = [i for i in mcolors.CSS4_COLORS if i not in [x,y,z]]
                ui.update_select(
                    id="VarQualActifTextColor",
                    label="Variables qualitatives",
                    choices={x : x for x in Colors},
                    selected="black"
                )

            #-------------------------------------------------------------------------------------------
            @output
            @render.ui
            def VarQuantTextChoice():
                if model.quanti_sup is not None:
                    return ui.TagList(
                        ui.input_select(
                            id="TextActifColor",
                            label=" actives",
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
            
            # Add Supplementary Quantitatives variables Conditional Panel
            @output
            @render.ui
            def VarQuantSupPanel():
                return ui.panel_conditional("input.choice == 'VarQuantSupRes'",
                            ui.br(),
                            ui.h5("Coordonnées"),
                            PanelConditional1(text="VarQuantSup",name="Coord"),
                            ui.hr(),
                            ui.h5("Cos2 - Qualité de la représentation"),
                            PanelConditional1(text="VarQuantSup",name="Cos2")
                        )
            
            # Add  supplementary variables/categories
            @output
            @render.ui
            def ModSupPanel():
                return ui.panel_conditional("input.choice == 'ModSupRes'",
                            ui.br(),
                            ui.h5("Coordonnées"),
                            PanelConditional1(text="ModSup",name="Coord"),
                            ui.hr(),
                            ui.h5("Cos2 - Qualité de la représentation"),
                            PanelConditional2(text="ModSup",name="Cos2"),
                            ui.hr(),
                            ui.h5("V-test"),
                            PanelConditional1(text="ModSup",name="Vtest")
                        )
            
            # Add  supplementary categoriacl variables panel
            @output
            @render.ui
            def VarQualSupPanel():
                return ui.panel_conditional("input.choice == 'VarQualSupRes'",
                            ui.br(),
                            ui.h5("Rapport de corrélation"),
                            PanelConditional1(text="VarQualSup",name="Eta2"),
                            ui.hr(),
                            ui.h5("Cos2 - Qualité de la représentation"),
                            PanelConditional1(text="VarQualSup",name="Cos2")
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
                        fig = fviz_famd_ind(self=model,
                                            axis=[int(input.Axis1()),int(input.Axis2())],
                                            color=input.IndTextActifColor(),
                                            color_sup = input.IndTextSupColor(),
                                            text_size = input.IndTextSize(),
                                            lim_contrib =input.IndLimContrib(),
                                            lim_cos2 = input.IndLimCos2(),
                                            title = input.IndTitle(),
                                            quali_sup=False,
                                            repel=input.IndPlotRepel(),
                                            ggtheme=pn.theme_gray())
                    else:
                        fig = fviz_famd_ind(self=model,
                                            axis=[int(input.Axis1()),int(input.Axis2())],
                                            color=input.IndTextActifColor(),
                                            text_size = input.IndTextSize(),
                                            lim_contrib =input.IndLimContrib(),
                                            lim_cos2 = input.IndLimCos2(),
                                            title = input.IndTitle(),
                                            quali_sup=False,
                                            repel=input.IndPlotRepel(),
                                            ggtheme=pn.theme_gray())
                    return fig
                elif input.IndTextColor() in ["cos2","contrib"]:
                    return  fviz_famd_ind(self=model,
                                          axis=[int(input.Axis1()),int(input.Axis2())],
                                          color=input.IndTextColor(),
                                          text_size = input.IndTextSize(),
                                          lim_contrib =input.IndLimContrib(),
                                          lim_cos2 = input.IndLimCos2(),
                                          title = input.IndTitle(),
                                          quali_sup=False,
                                          repel=input.IndPlotRepel(),
                                          ggtheme=pn.theme_gray())
                elif input.IndTextColor() == "varqual":
                    return fviz_famd_ind(self=model,
                                         axis=[int(input.Axis1()),int(input.Axis2())],
                                         text_size = input.IndTextSize(),
                                         lim_contrib =input.IndLimContrib(),
                                         lim_cos2 = input.IndLimCos2(),
                                         title = input.IndTitle(),
                                         habillage= input.IndTextVarQualColor(),
                                         add_ellipses=input.IndAddEllipse(),
                                         quali_sup=False,
                                         repel=input.IndPlotRepel(),
                                         ggtheme=pn.theme_gray())
                elif  input.IndTextColor() == "varquant":
                    return fviz_famd_ind(self=model,
                                         axis=[int(input.Axis1()),int(input.Axis2())],
                                         color=input.IndTextVarQuantColor(),
                                         text_size = input.IndTextSize(),
                                         lim_contrib =input.IndLimContrib(),
                                         lim_cos2 = input.IndLimCos2(),
                                         title = input.IndTitle(),
                                         habillage= None,
                                         add_ellipses=False,
                                         quali_sup=False,
                                         repel=input.IndPlotRepel(),
                                         ggtheme=pn.theme_gray())

            # ------------------------------------------------------------------------------
            # Individual Factor Map - PCA
            @output
            @render.plot(alt="Individuals Factor Map - FAMD")
            def RowFactorMap():
                return RowPlot().draw()
            
            # import io
            # @session.download(filename="Individuals-Factor-Map.png")
            # def IndGraphDownloadPng():
            #     with io.BytesIO() as buf:
            #         plt.savefig(RowPlot(), format="png")
            #         yield buf.getvalue()

            #####################################################################################
            #   Variables Categories Plot
            #------------------------------------------------------------------------------------
             #############################################################################################
            #  Variables/categories Factor Map
            ##############################################################################################

            @reactive.Calc
            def ModFactorPlot():
                if input.ModTextColor() == "actif/sup":
                    if model.quali_sup is not None:
                        fig = fviz_famd_mod(self=model,
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
                        fig = fviz_famd_mod(self=model,
                                            axis=[int(input.Axis1()),int(input.Axis2())],
                                            title=input.ModTitle(),
                                            color=input.ModTextActifColor(),
                                            color_sup=None,
                                            text_size=input.ModTextSize(),
                                            lim_contrib = input.ModLimContrib(),
                                            lim_cos2 = input.ModLimCos2(),
                                            repel=input.ModPlotRepel(),
                                            ggtheme=pn.theme_gray())
                    return fig
                elif input.ModTextColor() in ["cos2","contrib"]:
                    if model.quali_sup is not None:
                        fig = fviz_famd_mod(self=model,
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
                        fig = fviz_famd_mod(self=model,
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
                

            # Variables categories Factor Map - FAMD
            @output
            @render.plot(alt="Variables categories Factor Map - FAMD")
            def ModFactorMap():
                return ModFactorPlot().draw()
            
            #################################################################################################
            # Variables Map
            #-------------------------------------------------------------------------------------------------
            @reactive.Calc
            def VarFactorPlot():
                if (model.quali_sup is not None) and (model.quanti_sup is not None):
                    return fviz_famd_var(self=model,
                                        axis=[int(input.Axis1()),int(input.Axis2())],
                                        title=input.VarTitle(),
                                        color_quali=input.VarQualActifTextColor(),
                                        color_quanti=input.VarQuantActifTextColor(),
                                        color_quali_sup=input.VarQualSupTextColor(),
                                        color_quanti_sup=input.VarQuantSupTextColor(),
                                        text_size=input.VarTextSize(),
                                        repel=input.VarPlotRepel(),
                                        ggtheme=pn.theme_gray())
                elif model.quali_sup is not None:
                    return fviz_famd_var(self=model,
                                        axis=[int(input.Axis1()),int(input.Axis2())],
                                        title=input.VarTitle(),
                                        color_quali=input.VarQualActifTextColor(),
                                        color_quanti=input.VarQuantActifTextColor(),
                                        color_quali_sup=input.VarQualSupTextColor(),
                                        text_size=input.VarTextSize(),
                                        repel=input.VarPlotRepel(),
                                        ggtheme=pn.theme_gray())
                elif model.quanti_sup is not None:
                    return fviz_famd_var(self=model,
                                        axis=[int(input.Axis1()),int(input.Axis2())],
                                        title=input.VarTitle(),
                                        color_quali=input.VarQualActifTextColor(),
                                        color_quanti=input.VarQuantActifTextColor(),
                                        color_quali_sup=input.VarQualSupTextColor(),
                                        color_quanti_sup=input.VarQuantSupTextColor(),
                                        text_size=input.VarTextSize(),
                                        repel=input.VarPlotRepel(),
                                        ggtheme=pn.theme_gray())
                else:
                    return fviz_famd_var(self=model,
                                        axis=[int(input.Axis1()),int(input.Axis2())],
                                        title=input.VarTitle(),
                                        color_quali=input.VarQualActifTextColor(),
                                        color_quanti=input.VarQuantActifTextColor(),
                                        text_size=input.VarTextSize(),
                                        repel=input.VarPlotRepel(),
                                        ggtheme=pn.theme_gray())

            # Variables Factor Map - MCA
            @output
            @render.plot(alt="Variables Factor Map - FAMD")
            def VarFactorMap():
                return VarFactorPlot().draw()
            
            #################################################################################################
            #   Supplementary Continuous variables
            #-------------------------------------------------------------------------------------------------
            # Supplementary Continuous variables MAP
            @reactive.Calc
            def VarQuantFactorPlot():
                if input.VarQuantTextColor() == "actif/sup":
                    if model.quanti_sup is not None:
                        return fviz_famd_col(self=model,
                                            axis=[int(input.Axis1()),int(input.Axis2())],
                                            title=input.VarQuantTitle(),
                                            color=input.VarQuantActifTextColor(),
                                            color_sup=input.VarQuantSupTextColor(),
                                            text_size=input.VarQuantTextSize(),
                                            lim_contrib=input.VarQuantLimContrib(),
                                            lim_cos2=input.VarQuantLimCos2(),
                                            ggtheme=pn.theme_gray())
                    else:
                        return fviz_famd_col(self=model,
                                            axis=[int(input.Axis1()),int(input.Axis2())],
                                            title=input.VarQuantTitle(),
                                            color=input.VarQuantActifTextColor(),
                                            text_size=input.VarQuantTextSize(),
                                            lim_contrib=input.VarQuantLimContrib(),
                                            lim_cos2=input.VarQuantLimCos2(),
                                            ggtheme=pn.theme_gray())
                else:
                    return pn.ggplot()
            
            @output
            @render.plot(alt="Quantitatives variables - FAMD")
            def VarQuantFactorMap():
                return VarQuantFactorPlot().draw()

            ##################################################################################################
            #   Tab Valeurs
            ##################################################################################################
            #-------------------------------------------------------------------------------------------
            # Eigenvalue - Scree plot
            @output
            @render.plot(alt="Scree Plot - FAMD")
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
                return DataTable(data=match_datalength(EigenData,input.EigenLen()),
                                filters=input.EigenFilter())
            
            #################################################################################################
            #   Categories/modalités
            #####################################################################################################
            #---------------------------------------------------------------------------------------------
            # Variables Coordinates
            @output
            @render.data_frame
            def ModCoordTable():
                ModCoord = model.quali_var_["coord"].round(4).reset_index()
                ModCoord.columns = ["Categories",*ModCoord.columns[1:]]
                return DataTable(data=match_datalength(data=ModCoord,value=input.ModCoordLen()),
                                filters=input.ModCoordFilter())
            
            #----------------------------------------------------------------------------------------------------
            # Variables Contributions
            @output
            @render.data_frame
            def ModContribTable():
                ModContrib = model.quali_var_["contrib"].round(4).reset_index()
                ModContrib.columns = ["Categories",*ModContrib.columns[1:]]
                return  DataTable(data=match_datalength(data=ModContrib,value=input.ModContribLen()),filters=input.ModContribFilter())
            
            #-----------------------------------------------------------------------------------------------------
            # Add Variables Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.ModContribGraphBtn)
            def _():
                GraphModalShow(text="Mod",name="Contrib",max_axis=model.call_["n_components"])
            
            @reactive.Calc
            def ModContribMap():
                return fviz_contrib(self=model,
                                    choice="quali_var",
                                    axis=input.ModContribAxis(),
                                    top_contrib=int(input.ModContribTop()),
                                    color=input.ModContribColor(),
                                    bar_width=input.ModContribBarWidth(),
                                    ggtheme=pn.theme_gray())

            # Plot variables Contributions
            @output
            @render.plot(alt="Variables/categories contributions Map - FAMD")
            def ModContribPlot():
                return ModContribMap().draw()
            
            #-----------------------------------------------------------------------------------------------------------
            # Variables categories Cos2 
            @output
            @render.data_frame
            def ModCos2Table():
                ModCos2 = model.quali_var_["cos2"].round(4).reset_index()
                ModCos2.columns = ["Categories",*ModCos2.columns[1:]]
                return  DataTable(data=match_datalength(data=ModCos2,value=input.ModCos2Len()),filters=input.ModCos2Filter())
            
            #-------------------------------------------------------------------------------------------------------------
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.ModCos2GraphBtn)
            def _():
                GraphModalShow(text="Mod",name="Cos2")
            
            @reactive.Calc
            def ModCos2Map():
                return fviz_cos2(self=model,
                                choice="quali_var",
                                axis=input.ModCos2Axis(),
                                top_cos2=int(input.ModCos2Top()),
                                color=input.ModCos2Color(),
                                bar_width=input.ModCos2BarWidth(),
                                ggtheme=pn.theme_gray())

            # Plot variables categories Cos2
            @output
            @render.plot(alt="Variables/categories Cosines Map - FAMD")
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
                IndCoord.columns = ["Individus",*IndCoord.columns[1:]]
                return DataTable(data = match_datalength(IndCoord,input.IndCoordLen()),filters=input.IndCoordFilter())
            
            # Individuals Contributions
            @output
            @render.data_frame
            def IndContribTable():
                IndContrib = model.ind_["contrib"].round(4).reset_index()
                IndContrib.columns = ["Individus",*IndContrib.columns[1:]]
                return  DataTable(data=match_datalength(IndContrib,input.IndContribLen()),filters=input.IndContribFilter())
            
            # Add indiviuals Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.IndContribGraphBtn)
            def _():
                GraphModalShow(text="Ind",name="Contrib",max_axis=model.call_["n_components"])

            # Plot Individuals Contributions
            @output
            @render.plot(alt="Individuals Contributions Map - FAMD")
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
                IndCos2.columns = ["Individus",*IndCos2.columns[1:]]
                return  DataTable(data = match_datalength(IndCos2,input.IndCos2Len()),filters=input.IndCos2Filter())
            
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.IndCos2GraphBtn)
            def _():
                GraphModalShow(text="Ind",name="Cos2")

            # Plot variables Cos2
            @output
            @render.plot(alt="Individuals Cosines Map - FAMD")
            def IndCos2Plot():
                IndCos2Fig = fviz_cos2(self=model,
                                        choice="ind",
                                        axis=input.IndCos2Axis(),
                                        top_cos2=int(input.IndCos2Top()),
                                        color=input.IndCos2Color(),
                                        bar_width=input.IndCos2BarWidth(),
                                        ggtheme=pn.theme_gray())
                return IndCos2Fig.draw()

            ##########################################################################################
            # Conntinuous variables
            #-----------------------------------------------------------------------------------------
            # continuous variables coordinates
            @output
            @render.data_frame
            def VarQuantCoordTable():
                VarQuantCoord = model.quanti_var_["coord"].round(4).reset_index()
                VarQuantCoord.columns = ["Variables",*VarQuantCoord.columns[1:]]
                return  DataTable(data = match_datalength(VarQuantCoord,input.VarQuantCoordLen()),filters=input.VarQuantCoordFilter())
            
            # Continuous variables contributions
            @output
            @render.data_frame
            def VarQuantContribTable():
                VarQuantContrib = model.quanti_var_["contrib"].round(4).reset_index()
                VarQuantContrib.columns = ["Variables",*VarQuantContrib.columns[1:]]
                return  DataTable(data=match_datalength(VarQuantContrib,input.VarQuantContribLen()),filters=input.VarQuantContribFilter())
            
            # Variables Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.VarQuantContribGraphBtn)
            def _():
                GraphModalShow(text="VarQuant",name="Contrib")

            # Plot Individuals Contributions
            @output
            @render.plot(alt="Quantitative variables contributions Map - FAMD")
            def VarQuantContribPlot():
                VarQuantContribFig = fviz_contrib(self=model,
                                                  choice="quanti_var",
                                                  axis=input.VarQuantContribAxis(),
                                                  top_contrib=int(input.VarQuantContribTop()),
                                                  color = input.VarQuantContribColor(),
                                                  bar_width= input.VarQuantContribBarWidth(),
                                                  ggtheme=pn.theme_gray())
                return VarQuantContribFig.draw()
            
            # Supplementary continuous variables cos2
            @output
            @render.data_frame
            def VarQuantCos2Table():
                VarQuantCos2 = model.quanti_var_["cos2"].round(4).reset_index()
                VarQuantCos2.columns = ["Variables",*VarQuantCos2.columns[1:]]
                return  DataTable(data = match_datalength(VarQuantCos2,input.VarQuantCos2Len()),filters=input.VarQuantCos2Filter())
            
            # Variables Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.VarQuantCos2GraphBtn)
            def _():
                GraphModalShow(text="VarQuant",name="Cos2")

            # Plot Individuals Contributions
            @output
            @render.plot(alt="Quantitative variables cosinus Map - FAMD")
            def VarQuantCos2Plot():
                VarQuantCos2Fig = fviz_cos2(self=model,
                                            choice="quanti_var",
                                            axis=input.VarQuantCos2Axis(),
                                            top_cos2=int(input.VarQuantCos2Top()),
                                            color = input.VarQuantCos2Color(),
                                            bar_width= input.VarQuantCos2BarWidth(),
                                            ggtheme=pn.theme_gray())
                return VarQuantCos2Fig.draw()
            
            ############################################################################################
            #   Categorical Variables informations
            #------------------------------------------------------------------------------------------
            # Categorical Variables - Correlation Ratio
            @output
            @render.data_frame
            def VarQualEta2Table():
                VarQualEta2 = model.var_["coord"].drop(index=model.quanti_var_["coord"].index.tolist()).round(4).reset_index()
                VarQualEta2.columns = ["Variables",*VarQualEta2.columns[1:]]
                return DataTable(data = match_datalength(VarQualEta2,input.VarQualEta2Len()),filters=input.VarQualEta2Filter())
            
            # Categorical variables - Contributions
            @output
            @render.data_frame
            def VarQualContribTable():
                VarQualContrib = model.var_["contrib"].drop(index=model.quanti_var_["coord"].index.tolist()).round(4).reset_index()
                VarQualContrib.columns = ["Variables",*VarQualContrib.columns[1:]]
                return  DataTable(data=match_datalength(VarQualContrib,input.VarQualContribLen()),filters=input.VarQualContribFilter())
            
            # Categorical variables - Cosinus
            @output
            @render.data_frame
            def VarQualCos2Table():
                VarQualCos2 = model.var_["cos2"].drop(index=model.quanti_var_["coord"].index.tolist()).round(4).reset_index()
                VarQualCos2.columns = ["Variables",*VarQualCos2.columns[1:]]
                return  DataTable(data=match_datalength(VarQualCos2,input.VarQualCos2Len()),filters=input.VarQualCos2Filter())
            
            ############################################################################################
            #    Supplementary individuals informations
            #-------------------------------------------------------------------------------------------
            # Supplementary Individual- Coordinates
            @output
            @render.data_frame
            def IndSupCoordTable():
                IndSupCoord = model.ind_sup_["coord"].round(4).reset_index()
                IndSupCoord.columns = ["Individus",*IndSupCoord.columns[1:]]
                return  DataTable(data = match_datalength(IndSupCoord,input.IndSupCoordLen()),filters=input.IndSupCoordFilter())
            
            # Supplementary Individual - Cos2
            @output
            @render.data_frame
            def IndSupCos2Table():
                IndSupCos2 = model.ind_sup_["cos2"].round(4).reset_index()
                IndSupCos2.columns = ["Individus",*IndSupCos2.columns[1:]]
                return  DataTable(data = match_datalength(IndSupCos2,input.IndSupCos2Len()),filters=input.IndSupCos2Filter())
            
            ##########################################################################################
            # Supplementary continuous variables
            #-----------------------------------------------------------------------------------------
            # Supplementary continuous variables - coordinates
            @output
            @render.data_frame
            def VarQuantSupCoordTable():
                VarQuantSupCoord = model.quanti_sup_["coord"].round(4).reset_index()
                VarQuantSupCoord.columns = ["Variables",*VarQuantSupCoord.columns[1:]]
                return  DataTable(data = match_datalength(VarQuantSupCoord,input.VarQuantSupCoordLen()),filters=input.VarQuantSupCoordFilter())
            
            # Supplementary continuous variables cos2
            @output
            @render.data_frame
            def VarQuantSupCos2Table():
                VarQuantSupCos2 = model.quanti_sup_["cos2"].round(4).reset_index()
                VarQuantSupCos2.columns = ["Variables",*VarQuantSupCos2.columns[1:]]
                return  DataTable(data = match_datalength(VarQuantSupCos2,input.VarQuantSupCos2Len()),filters=input.VarQuantSupCos2Filter())

            ###########################################################################################
            # Supplementary variables categories
            #-------------------------------------------------------------------------------------------
            #-------------------------------------------------------------------------------------
            ## Supplementary Variables/categories coordinates
            @output
            @render.data_frame
            def ModSupCoordTable():
                ModSupCoord = model.quali_sup_["coord"].round(4).reset_index()
                ModSupCoord.columns = ["Categories",*ModSupCoord.columns[1:]]
                return DataTable(data=match_datalength(data=ModSupCoord,value=input.ModSupCoordLen()),filters=input.ModSupCoordFilter())
            
            #-------------------------------------------------------------------------------
            # Supplementary variables/categories Cosinus
            @output
            @render.data_frame
            def ModSupCos2Table():
                ModSupCos2 = model.quali_sup_["cos2"].round(4).reset_index()
                ModSupCos2.columns = ["Categories",*ModSupCos2.columns[1:]]
                return DataTable(data=match_datalength(data=ModSupCos2,value=input.ModSupCos2Len()),filters=input.ModSupCos2Filter())
            
            #-------------------------------------------------------------------------------
            # Supplementary variables/categories Vtest
            @output
            @render.data_frame
            def ModSupVtestTable():
                ModSupVtest = model.quali_sup_["vtest"].round(4).reset_index()
                ModSupVtest.columns = ["Categories",*ModSupVtest.columns[1:]]
                return DataTable(data=match_datalength(data=ModSupVtest,value=input.ModSupVtestLen()),filters=input.ModSupVtestFilter())
            
            ############################################################################################
            #   Supplementary Categorical Variables informations
            #------------------------------------------------------------------------------------------
            # Supplementary Categorical Variables - Correlation Ratio
            @output
            @render.data_frame
            def VarQualSupEta2Table():
                VarQualSupEta2 = model.quali_sup_["eta2"].round(4).reset_index()
                VarQualSupEta2.columns = ["Variables",*VarQualSupEta2.columns[1:]]
                return DataTable(data = match_datalength(VarQualSupEta2,input.VarQualSupEta2Len()),filters=input.VarQualSupEta2Filter())
            
            # Supplementary Categorical variables - Cosinus
            @output
            @render.data_frame
            def VarQualSupCos2Table():
                VarQualSupCos2 = model.quali_sup_["cos2"].round(4).reset_index()
                VarQualSupCos2.columns = ["Variables",*VarQualSupCos2.columns[1:]]
                return  DataTable(data=match_datalength(VarQualSupCos2,input.VarQualSupCos2Len()),filters=input.VarQualSupCos2Filter())
            
            ############################################################################################
            #    Tab : Description automatique des axes
            ############################################################################################
            #--------------------------------------------------------------------------------------------------------
            @output
            @render.data_frame
            def Dim1DescTable():
                DimDesc = dimdesc(self=model,axis=None,proba=float(input.pvalueDimdesc()))
                if isinstance(DimDesc[input.Dimdesc()],dict):
                    DimDescQuanti = DimDesc[input.Dimdesc()]["quanti"].reset_index().rename(columns={"index":"Variables"})
                elif isinstance(DimDesc[input.Dimdesc()],pd.DataFrame):
                    DimDescQuanti = DimDesc[input.Dimdesc()].reset_index().rename(columns={"index":"Variables"})
                else:
                    DimDescQuanti = pd.DataFrame()
                return  DataTable(data = match_datalength(DimDescQuanti,input.Dim1DescLen()),filters=input.Dim1DescFilter())
            
            #--------------------------------------------------------------------------------------------------
            @output
            @render.data_frame
            def Dim2DescTable():
                DimDesc = dimdesc(self=model,axis=None,proba=float(input.pvalueDimdesc()))
                if isinstance(DimDesc[input.Dimdesc()],dict):
                    DimDescQuali = DimDesc[input.Dimdesc()]["quali"].reset_index().rename(columns={"index":"Variables"})
                elif isinstance(DimDesc[input.Dimdesc()],pd.DataFrame):
                    DimDescQuali = DimDesc[input.Dimdesc()].reset_index().rename(columns={"index":"Variables"})
                else:
                    DimDescQuali = pd.DataFrame()
                return  DataTable(data = match_datalength(DimDescQuali,input.Dim2DescLen()),filters=input.Dim2DescFilter())
            
            ###########################################################################################
            #   Tab : Resumé du jeu de données
            ###########################################################################################
            #-----------------------------------------------------------------------------------------------
            ### Statistiques descriptives
            # @output
            # @render.data_frame
            # def StatsDescTable():
            #     data = model.active_data_[fa_model.quanti_labels_]
            #     if fa_model.quanti_sup_labels_ is not None:
            #         quanti_sup = fa_model.data_[fa_model.quanti_sup_labels_]
            #         data = pd.concat([data,quanti_sup],axis=1)
            #         if fa_model.row_sup_labels_ is not None:
            #             data = data.drop(index=fa_model.row_sup_labels_)

            #     StatsDesc = data.describe(include="all").round(4).T.reset_index().rename(columns={"index":"Variables"})
            #     return  DataTable(data = match_datalength(StatsDesc,input.StatsDescLen()),
            #                     filters=input.StatsDescFilter())
            
            # Histogramme
            # @output
            # @render.plot(alt="Hist - Plot")
            # def VarHistGraph():
            #     data = fa_model.active_data_[fa_model.quanti_labels_]
            #     if fa_model.quanti_sup_labels_ is not None:
            #         quanti_sup = fa_model.data_[fa_model.quanti_sup_labels_]
            #         data = pd.concat([data,quanti_sup],axis=1)
            #         if fa_model.row_sup_labels_ is not None:
            #             data = data.drop(index=fa_model.row_sup_labels_)
            #     # Initialise
            #     p = pn.ggplot(data,pn.aes(x=input.VarQuantLabel()))
            #     # Add density
            #     if input.AddDensity():
            #         p = (p + pn.geom_histogram(pn.aes(y="..density.."), color="darkblue", fill="lightblue")+
            #             pn.geom_density(alpha=.2, fill="#FF6666"))
            #     else:
            #         p = p + pn.geom_histogram(color="darkblue", fill="lightblue")
                
            #     p = p + pn.ggtitle(f"Histogram de {input.VarQuantLabel()}")

            #     return p.draw()

            # Matrice des corrélations
            # @output
            # @render.data_frame
            # def CorrMatrixTable():
            #     data = fa_model.active_data_[fa_model.quanti_labels_]
            #     if fa_model.quanti_sup_labels_ is not None:
            #         quanti_sup = fa_model.data_[fa_model.quanti_sup_labels_]
            #         data = pd.concat([data,quanti_sup],axis=1)
            #         if fa_model.row_sup_labels_ is not None:
            #             data = data.drop(index=fa_model.row_sup_labels_)
                
            #     corr_mat = data.corr(method="pearson").round(4).reset_index().rename(columns={"index":"Variables"})
            #     return DataTable(data = match_datalength(corr_mat,input.CorrMatrixLen()),
            #                     filters=input.CorrMatrixFilter())

            #---------------------------------------------------------------------------------------------
            # Diagramme en barres
            # @output
            # @render.plot(alt="Bar-Plot")
            # def VarBarPlotGraph():
            #     data = fa_model.active_data_[fa_model.quali_labels_]
            #     if fa_model.quali_sup_labels_ is not None:
            #         quali_sup = fa_model.data_[fa_model.quali_sup_labels_]
            #         data = pd.concat([data,quali_sup],axis=1)
            #         if fa_model.row_sup_labels_ is not None:
            #             data = data.drop(index=fa_model.row_sup_labels_)
            #     p = pn.ggplot(data,pn.aes(x=input.VarQualLabel()))+ pn.geom_bar(color="darkblue", fill="lightblue")
            #     return p.draw()
            
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
        """
        
        """

        nest_asyncio.apply()
        uvicorn.run(self.run(**kwargs))
    
    def stop(self):
        """
        
        
        """
        app = App(ui=self.app_ui, server=self.server)
        return app.stop()


