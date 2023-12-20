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
from scientisttools.ggplot import (
    fviz_mca_ind,
    fviz_mca_mod,
    fviz_mca_var,
    fviz_eig, 
    fviz_contrib,
    fviz_cosines,
    fviz_corrplot,
    fviz_corrcircle)
from scientisttools.extractfactor import (
    get_eig,
    get_mca_ind,
    get_mca_var,
    get_mca_mod,
    dimdesc)

from scientistshiny.function import *

colors = mcolors.CSS4_COLORS
colors["cos2"] = "cos2"
colors["contrib"] = "contrib"

css_path = Path(__file__).parent / "www" / "style.css"

class MCAshiny:
    """
    Multiple Correspondance Analysis (MCA) with scientistshiny

    Description
    -----------
    Performs Multiple Correspondance Analysis (PCA) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables on a Shiny application.
    Graphics can be downloaded in png, jpg and pdf.

    Usage
    -----
    PCAshiny(fa_model)

    Parameters:
    ----------
    fa_model : An instance of class MCA. A MCA result from scientisttools.

    Returns:
    -------
    Graphs : a tab containing the individuals factor map and the variables factor map
    Values : a tab containing the summary of the PCA performed, the eigenvalue, the results
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
    > from scientisttools.decomposition import PCA
    > from scientistshiny import PCAshiny


    for jupyter notebooks
    https://stackoverflow.com/questions/74070505/how-to-run-fastapi-application-inside-jupyter
    """


    def __init__(self,fa_model=None):
        if fa_model.model_ != "mca":
            raise ValueError("Error : 'fa_model' must be an instance of class MCA")
        
        # -----------------------------------------------------------------------------------
        # Initialise value choice
        value_choice = {"EigenRes" : "Valeurs propres",
                        "ModRes"   : "Résultats des modalités",
                        "IndRes"   : "Résultats sur les individus",
                        "VarRes"   : "Résultats sur les variables"}
        if fa_model.row_sup_labels_ is not None:
            value_choice.update({"IndSupRes" : "Résultats des individus supplémentaires"})
        if fa_model.quanti_sup_labels_ is not None:
            value_choice.update({"VarQuantRes" : "Résultats des variables quantitatives supplémentaires"})
        if fa_model.quali_sup_labels_ is not None:
            value_choice.update({"VarSupRes" : "Résultats des variables qualitatives supplémentaires"})
        
        # Plot Choice
        PlotChoice = {"IndPlot":"individus",
                      "ModPlot":"modalités",
                      "VarPlot":"Variables"}
        if fa_model.quanti_sup_labels_ is not None:
            PlotChoice.update({"VarQuantPlot" : "Variables quantitatives"})
        
        # Dim Desc Choice
        DimDescChoice = {}
        for i in range(min(3,fa_model.n_components_)):
            DimDescChoice.update({"Dim."+str(i+1) : "Dimension "+str(i+1)})
        
        # Add Supplementary Qualitatives Variables
        VarLabelChoice = list(fa_model.original_data_.columns)
        if fa_model.quali_sup_labels_ is not None:
            for i in range(len(fa_model.quali_sup_labels_)):
                VarLabelChoice.insert(len(VarLabelChoice)+1,fa_model.quali_sup_labels_[i])

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
                            choices={x:x for x in range(fa_model.n_components_)},
                            selected=0,
                            multiple=False
                        ),
                        style="display: inline-block;"
                    ),
                    ui.div(
                        ui.input_select(
                            id="Axis2",
                            label="",
                            choices={x:x for x in range(fa_model.n_components_)},
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
                        ui.input_select(
                            id="VarPointSelect",
                            label="Libellés des points pour",
                            choices={
                                "none"    : "Pas de sélection",
                                "cos2"    : "Cosinus"
                                },
                            selected="none",
                            multiple=False,
                            width="100%"
                        ),
                        ui.panel_conditional("input.VarPointSelect === 'cos2'",
                            ui.div(ui.input_slider(id="VarLimCos2", label = "Libellés pour un cos2 plus grand que",min = 0, max = 1,value=0,step=0.05),align="center")              
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
                    ui.nav("Graphes",
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
                    ui.nav("Valeurs",
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
                            ui.h5("Cos2 - Qualité de la représentation"),
                            PanelConditional1(text="Var",name="Cos2")
                        ),
                        ui.output_ui("IndSupPanel"),
                        ui.output_ui("VarSupPanel"),  
                        ui.output_ui("VarQuantPanel")
                    ),
                    ui.nav("Description automatique des axes",
                        ui.row(
                            ui.column(7,ui.input_radio_buttons(id="pvalueDimdesc",label="Probabilité critique",choices={x:y for x,y in zip([0.01,0.05,0.1,1.0],["Significance level 1%","Significance level 5%","Significance level 10%","None"])},selected=0.05,width="100%",inline=True)),
                            ui.column(5,ui.input_radio_buttons(id="Dimdesc",label="Choisir les dimensions",choices=DimDescChoice,selected="Dim.1",inline=True))
                        ),
                        ui.output_ui(id="DimDesc")
                    ),
                    ui.nav("Résumé du jeu de données",
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
                        ui.panel_conditional("input.ResumeChoice==='StatsDesc'",
                            PanelConditional1(text="StatsDesc",name="")
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
                    ui.nav("Données",
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
                Dim = [i for i in range(fa_model.n_components_) if i > x]
                ui.update_select(
                    id="Axis2",
                    label="",
                    choices={x : x for x in Dim},
                    selected=Dim[0]
                )
            
            @reactive.Effect
            def _():
                x = int(input.Axis2())
                Dim = [i for i in range(fa_model.n_components_) if i < x]
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
                if fa_model.row_sup_labels_ is not None:
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
                if fa_model.quanti_sup_labels_ is not None:
                    return ui.TagList(
                        ui.input_select(
                            id="IndTextVarQuantColor",
                            label="Choix de la variable",
                            choices={x:x for x in fa_model.quanti_sup_labels_},
                            selected=fa_model.quanti_sup_labels_[0],
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
                if fa_model.quali_sup_labels_ is not None:
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
                if fa_model.quali_sup_labels_ is not None and fa_model.quanti_sup_labels_ is not None:
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
                elif fa_model.quali_sup_labels_ is not None:
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
                elif fa_model.quanti_sup_labels_ is not None:
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
                if fa_model.quanti_sup_labels_ is not None:
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
                            PanelConditional2(text="IndSup",name="Cos2") 
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
                            PanelConditional2(text="VarSup",name="Cos2")
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
                    if fa_model.row_sup_labels_ is not None:
                        fig = fviz_mca_ind(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            color=input.IndTextActifColor(),
                            color_sup = input.IndTextSupColor(),
                            text_size = input.IndTextSize(),
                            lim_contrib =input.IndLimContrib(),
                            lim_cos2 = input.IndLimCos2(),
                            title = input.IndTitle(),
                            repel=input.IndPlotRepel()
                        )
                    else:
                        fig = fviz_mca_ind(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            color=input.IndTextActifColor(),
                            text_size = input.IndTextSize(),
                            lim_contrib =input.IndLimContrib(),
                            lim_cos2 = input.IndLimCos2(),
                            title = input.IndTitle(),
                            repel=input.IndPlotRepel()
                        )
                elif input.IndTextColor() in ["cos2","contrib"]:
                    fig = fviz_mca_ind(
                        self=fa_model,
                        axis=[int(input.Axis1()),int(input.Axis2())],
                        color=input.IndTextColor(),
                        text_size = input.IndTextSize(),
                        lim_contrib =input.IndLimContrib(),
                        lim_cos2 = input.IndLimCos2(),
                        title = input.IndTitle(),
                        repel=input.IndPlotRepel()
                    )
                elif input.IndTextColor() == "varqual":
                    fig = fviz_mca_ind(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            text_size = input.IndTextSize(),
                            lim_contrib =input.IndLimContrib(),
                            lim_cos2 = input.IndLimCos2(),
                            title = input.IndTitle(),
                            habillage= input.IndTextVarQualColor(),
                            add_ellipse=input.IndAddEllipse(),
                            repel=input.IndPlotRepel()
                        )
                elif  input.IndTextColor() == "varquant":
                    if fa_model.quanti_sup_labels_ is not None:
                        fig = fviz_mca_ind(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            color=input.IndTextVarQuantColor(),
                            text_size = input.IndTextSize(),
                            lim_contrib =input.IndLimContrib(),
                            lim_cos2 = input.IndLimCos2(),
                            title = input.IndTitle(),
                            habillage= None,
                            add_ellipse=input.IndAddEllipse(),
                            repel=input.IndPlotRepel()
                        )
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
                    if fa_model.quali_sup_labels_ is not None:
                        fig = fviz_mca_mod(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            title=input.ModTitle(),
                            color=input.ModTextActifColor(),
                            color_sup=input.ModTextSupColor(),
                            text_size=input.ModTextSize(),
                            lim_contrib = input.ModLimContrib(),
                            lim_cos2 = input.ModLimCos2(),
                            repel=input.ModPlotRepel()
                            )
                    else:
                        fig = fviz_mca_mod(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            title=input.ModTitle(),
                            color=input.ModTextActifColor(),
                            color_sup=None,
                            text_size=input.ModTextSize(),
                            lim_contrib = input.ModLimContrib(),
                            lim_cos2 = input.ModLimCos2(),
                            repel=input.ModPlotRepel()
                            )
                elif input.ModTextColor() in ["cos2","contrib"]:
                    if fa_model.quali_sup_labels_ is not None:
                        fig = fviz_mca_mod(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            title=input.ModTitle(),
                            color=input.ModTextColor(),
                            color_sup=input.ModTextSupColor(),
                            text_size=input.ModTextSize(),
                            lim_contrib = input.ModLimContrib(),
                            lim_cos2 = input.ModLimCos2(),
                            repel=input.ModPlotRepel()
                            )
                    else:
                        fig = fviz_mca_mod(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            title=input.ModTitle(),
                            color=input.ModTextColor(),
                            color_sup=None,
                            text_size=input.ModTextSize(),
                            lim_contrib = input.ModLimContrib(),
                            lim_cos2 = input.ModLimCos2(),
                            repel=input.ModPlotRepel()
                            )
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
                if (fa_model.quali_sup_labels_ is not None) and (fa_model.quanti_sup_labels_ is not None):
                    fig = fviz_mca_var(
                        self=fa_model,
                        axis=[int(input.Axis1()),int(input.Axis2())],
                        title=input.VarTitle(),
                        color=input.VarTextActifColor(),
                        color_sup=input.VarSupTextSupColor(),
                        color_quanti_sup=input.VarQuantTextColor(),
                        text_size=input.VarTextSize(),
                        lim_cos2 = input.VarLimCos2(),
                        repel=input.VarPlotRepel()
                        )
                elif fa_model.quali_sup_labels_ is not None:
                    fig = fviz_mca_var(
                        self=fa_model,
                        axis=[int(input.Axis1()),int(input.Axis2())],
                        title=input.VarTitle(),
                        color=input.VarTextActifColor(),
                        color_sup=input.VarSupTextSupColor(),
                        text_size=input.VarTextSize(),
                        lim_cos2 = input.VarLimCos2(),
                        repel=input.VarPlotRepel()
                        )
                elif fa_model.quanti_sup_labels_ is not None:
                    fig = fviz_mca_var(
                        self=fa_model,
                        axis=[int(input.Axis1()),int(input.Axis2())],
                        title=input.VarTitle(),
                        color=input.VarTextActifColor(),
                        color_quanti_sup=input.VarQuantTextColor(),
                        text_size=input.VarTextSize(),
                        lim_cos2 = input.VarLimCos2(),
                        repel=input.VarPlotRepel()
                        )
                else:
                    fig = fviz_mca_var(
                        self=fa_model,
                        axis=[int(input.Axis1()),int(input.Axis2())],
                        title=input.VarTitle(),
                        color=input.VarTextActifColor(),
                        text_size=input.VarTextSize(),
                        lim_cos2 = input.VarLimCos2(),
                        repel=input.VarPlotRepel()
                        )
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
                fig =  fviz_corrcircle(
                    self=fa_model,
                    axis=[int(input.Axis1()),int(input.Axis2())],
                    title=input.VarQuantTitle(),
                    text_size=input.VarQuantTextSize()
                )
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
                EigenFig = fviz_eig(self=fa_model,
                                    choice=input.EigenChoice(),
                                    add_labels=input.EigenLabel())
                return EigenFig.draw()
            
            # Eigen value - DataFrame
            @output
            @render.data_frame
            def EigenTable():
                EigenData = get_eig(fa_model).round(4).reset_index().rename(columns={"index":"dimensions"})
                EigenData.columns = [x.capitalize() for x in EigenData.columns]
                return DataTable(data=match_datalength(EigenData,input.EigenLen()),
                                filters=input.EigenFilter())
            
            #---------------------------------------------------------------------------------------------
            #################################################################################################
            #   Categories/modalités
            #####################################################################################################
            #---------------------------------------------------------------------------------------------
            # Variables Coordinates
            @output
            @render.data_frame
            def ModCoordTable():
                ModCoord = get_mca_mod(fa_model)["coord"].round(4).reset_index().rename(columns={"index" : "Categories"})
                return DataTable(data=match_datalength(data=ModCoord,value=input.ModCoordLen()),
                                filters=input.ModCoordFilter())
            
            #----------------------------------------------------------------------------------------------------
            # Variables Contributions
            @output
            @render.data_frame
            def ModContribTable():
                ModContrib = get_mca_mod(fa_model)["contrib"].round(4).reset_index().rename(columns={"index" : "Categories"})
                return  DataTable(data=match_datalength(data=ModContrib,value=input.ModContribLen()),
                                filters=input.ModContribFilter())
            
            #-----------------------------------------------------------------------------------------------------
            # Add Variables Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.ModContribGraphBtn)
            def _():
                GraphModalShow(text="Mod",name="Contrib")
            
            @reactive.Calc
            def ModContribMap():
                fig = fviz_contrib(
                    self=fa_model,
                    choice="mod",
                    axis=input.ModContribAxis(),
                    top_contrib=int(input.ModContribTop()),
                    color=input.ModContribColor(),
                    bar_width=input.ModContribBarWidth()
                    )
                return fig

            # Plot variables Contributions
            @output
            @render.plot(alt="Variables categories contributions Map - MCA")
            def ModContribPlot():
                return ModContribMap().draw()
            
            #----------------------------------------------------------------------------------------------------------------
            # Add Variables catgories contributions correlation Modal Show
            @reactive.Effect
            @reactive.event(input.ModContribCorrGraphBtn)
            def _():
                GraphModelModal2(text="Mod",name="Contrib",title=None)
            
            @reactive.Calc
            def ModContribCorrMap():
                ModContrib = get_mca_mod(fa_model)["contrib"]
                fig = fviz_corrplot(
                    X=ModContrib,
                    title=input.ModContribCorrTitle(),
                    outline_color=input.ModContribCorrColor(),
                    colors=[input.ModContribCorrLowColor(),
                            input.ModContribCorrMidColor(),
                            input.ModContribCorrHightColor()
                            ]
                    )+pn.theme_gray()
                return fig

            # Plot variables Contributions/correlations Map - PCA
            @output
            @render.plot(alt="Variables categories contributions/correlations Map - MCA")
            def ModContribCorrPlot():
                return ModContribCorrMap().draw()
            
            #-----------------------------------------------------------------------------------------------------------
            # Variables categories Cos2 
            @output
            @render.data_frame
            def ModCos2Table():
                ModCos2 = get_mca_mod(fa_model)["cos2"].round(4).reset_index().rename(columns={"index" : "Categories"})
                return  DataTable(data=match_datalength(data=ModCos2,value=input.ModCos2Len()),
                                filters=input.ModCos2Filter())
            
            #-------------------------------------------------------------------------------------------------------------
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.ModCos2GraphBtn)
            def _():
                GraphModalShow(text="Mod",name="Cos2")
            
            @reactive.Calc
            def ModCos2Map():
                fig = fviz_cosines(
                    self=fa_model,
                    choice="mod",
                    axis=input.ModCos2Axis(),
                    top_cos2=int(input.ModCos2Top()),
                    color=input.ModCos2Color(),
                    bar_width=input.ModCos2BarWidth())
                return fig

            # Plot variables categories Cos2
            @output
            @render.plot(alt="Variables categories Cosines Map - MCA")
            def ModCos2Plot():
                return ModCos2Map().draw()
            
            #----------------------------------------------------------------------------------------
            # Add Variables categories Cosinus Correlation Modal Show
            @reactive.Effect
            @reactive.event(input.ModCos2CorrGraphBtn)
            def _():
                GraphModelModal2(text="Mod",name="Cos2",title=None)
            
            @reactive.Calc
            def ModCos2CorrMap():
                ModCos2 = get_mca_mod(fa_model)["cos2"]
                fig = fviz_corrplot(
                    X=ModCos2,
                    title=input.ModCos2CorrTitle(),
                    outline_color=input.ModCos2CorrColor(),
                    colors=[input.ModCos2CorrLowColor(),
                            input.ModCos2CorrMidColor(),
                            input.ModCos2CorrHightColor()
                            ])+pn.theme_gray()
                return fig

            #--------------------------------------------------------------------------------------------------
            # Plot variables Contributions
            @output
            @render.plot(alt="Variables categories Contributions/Correlations Map - MCA")
            def ModCos2CorrPlot():
                return ModCos2CorrMap().draw()
            
            ########################################################################################################
            # Individuals informations
            #---------------------------------------------------------------------------------------------
            # Individuals Coordinates
            @output
            @render.data_frame
            def IndCoordTable():
                IndCoord = get_mca_ind(fa_model)["coord"].round(4).reset_index()
                return DataTable(data = match_datalength(IndCoord,input.IndCoordLen()),
                                filters=input.IndCoordFilter())
            
            # Individuals Contributions
            @output
            @render.data_frame
            def IndContribTable():
                IndContrib = get_mca_ind(fa_model)["contrib"].round(4).reset_index()
                return  DataTable(data=match_datalength(IndContrib,input.IndContribLen()),
                                filters=input.IndContribFilter())
            
            # Add indiviuals Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.IndContribGraphBtn)
            def _():
                GraphModalShow(text="Ind",name="Contrib")

            # Plot Individuals Contributions
            @output
            @render.plot(alt="Individuals Contributions Map - MCA")
            def IndContribPlot():
                IndContribFig = fviz_contrib(self=fa_model,
                                            choice="ind",
                                            axis=input.IndContribAxis(),
                                            top_contrib=int(input.IndContribTop()),
                                            color = input.IndContribColor(),
                                            bar_width= input.IndContribBarWidth())
                return IndContribFig.draw()
            
            # Add Variables Contributions Correlation Modal Show
            @reactive.Effect
            @reactive.event(input.IndContribCorrGraphBtn)
            def _():
                GraphModelModal2(text="Ind",name="Contrib",title=None)

            # Plot variables Contributions
            @output
            @render.plot(alt="Individuals Contributions/Correlations Map - MCA")
            def IndContribCorrPlot():
                IndContrib = get_mca_ind(fa_model)["contrib"]
                IndContribCorrFig = fviz_corrplot(X=IndContrib,
                                                title=input.IndContribCorrTitle(),
                                                outline_color=input.IndContribCorrColor(),
                                                colors=[input.IndContribCorrLowColor(),
                                                        input.IndContribCorrMidColor(),
                                                        input.IndContribCorrHightColor()
                                                        ])+pn.theme_gray()
                return IndContribCorrFig.draw()
            
            # Individuals Cos2 
            @output
            @render.data_frame
            def IndCos2Table():
                IndCos2 = get_mca_ind(fa_model)["cos2"].round(4).reset_index()
                return  DataTable(data = match_datalength(IndCos2,input.IndCos2Len()),
                                filters=input.IndCos2Filter())
            
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.IndCos2GraphBtn)
            def _():
                GraphModalShow(text="Ind",name="Cos2")

            # Plot variables Cos2
            @output
            @render.plot(alt="Individuals Cosines Map - MCA")
            def IndCos2Plot():
                IndCos2Fig = fviz_cosines(self=fa_model,
                                choice="ind",
                                axis=input.IndCos2Axis(),
                                top_cos2=int(input.IndCos2Top()),
                                color=input.IndCos2Color(),
                                bar_width=input.IndCos2BarWidth())
                return IndCos2Fig.draw()
            
            # Add Variables Cosines Correlation Modal Show
            @reactive.Effect
            @reactive.event(input.IndCos2CorrGraphBtn)
            def _():
                GraphModelModal2(text="Ind",name="Cos2",title=None)

            # Plot variables Contributions
            @output
            @render.plot(alt="Individuals Cosinus/Correlations Map - MCA")
            def IndCos2CorrPlot():
                IndCos2 = get_mca_ind(fa_model)["cos2"]
                IndCos2CorrFig = fviz_corrplot(X=IndCos2,
                                            title=input.IndCos2CorrTitle(),
                                            outline_color=input.IndCos2CorrColor(),
                                            colors=[input.IndCos2CorrLowColor(),
                                                    input.IndCos2CorrMidColor(),
                                                    input.IndCos2CorrHightColor()
                                                ])+pn.theme_gray()
                return IndCos2CorrFig.draw()
            
            ###########################################################################################
            #   Variables informations
            #------------------------------------------------------------------------------------------
            # Variables - Rapport de corrélation
            @output
            @render.data_frame
            def VarEta2Table():
                VarEta2 = get_mca_var(fa_model)["eta2"].round(4).reset_index()
                return DataTable(data = match_datalength(VarEta2,input.VarEta2Len()),
                                 filters=input.VarEta2Filter())
            
            # Variables Contributions
            @output
            @render.data_frame
            def VarCos2Table():
                VarCos2 = get_mca_var(fa_model)["cos2"].round(4).reset_index()
                return  DataTable(data=match_datalength(VarCos2,input.IndCos2Len()),
                                  filters=input.VarCos2Filter())
            
            ############################################################################################
            #    Supplementary individuals informations
            #-------------------------------------------------------------------------------------------
            # Supplementaru Individual Coordinates
            @output
            @render.data_frame
            def IndSupCoordTable():
                IndSupCoord = get_mca_ind(fa_model)["ind_sup"]["coord"].round(4).reset_index()
                return  DataTable(data = match_datalength(IndSupCoord,input.IndSupCoordLen()),
                                  filters=input.IndSupCoordFilter())
            
            # Supplementaru Individual Cos2
            @output
            @render.data_frame
            def IndSupCos2Table():
                IndSupCos2 = get_mca_ind(fa_model)["ind_sup"]["cos2"].round(4).reset_index()
                return  DataTable(data = match_datalength(IndSupCos2,input.IndSupCos2Len()),
                                  filters=input.IndSupCos2Filter())
            
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.IndSupCos2GraphBtn)
            def _():
                GraphModalShow(text="IndSup",name="Cos2")

            # Plot variables Cos2
            @output
            @render.plot(alt="Supplementary Individuals Cosines Map - PCA")
            def IndSupCos2Plot():
                IndSupCos2 = get_mca_ind(fa_model)["ind_sup"]["cos2"]
                IndSupCos2Fig = fviz_barplot(X=IndSupCos2,
                                            axis=input.IndSupCos2Axis(),
                                            top_corr=int(input.IndSupCos2Top()),
                                            color=input.IndSupCos2Color(),
                                            bar_width=input.IndSupCos2BarWidth(),
                                            ylabel="Supplementary individuals",
                                            xlabel="Cosinus",
                                            title=f"Cosinus of supplementary individuals to Dim-{input.IndSupCos2Axis()+1}")
                return IndSupCos2Fig.draw()
            
            # Add Variables Cosines Correlation Modal Show
            @reactive.Effect
            @reactive.event(input.IndSupCos2CorrGraphBtn)
            def _():
                GraphModelModal2(text="IndSup",name="Cos2",title=None)

            # Plot variables Contributions
            @output
            @render.plot(alt="Individuals Cosinus/Correlations Map - PCA")
            def IndSupCos2CorrPlot():
                IndSupCos2 = get_mca_ind(fa_model)["ind_sup"]["cos2"]
                IndSupCos2CorrFig = fviz_corrplot(X=IndSupCos2,
                                                title=input.IndSupCos2CorrTitle(),
                                                outline_color=input.IndSupCos2CorrColor(),
                                                colors=[input.IndSupCos2CorrLowColor(),
                                                        input.IndSupCos2CorrMidColor(),
                                                        input.IndSupCos2CorrHightColor()
                                                        ],
                                                    xlabel="Supplementary individuals")+pn.theme_gray()
                return IndSupCos2CorrFig.draw()
            
            ##########################################################################################
            # Supplementary continuous variables
            #-----------------------------------------------------------------------------------------
            # Supplementary continuous variables coordinates
            @output
            @render.data_frame
            def VarQuantCoordTable():
                VarQuantCoord = get_mca_var(fa_model)["quanti_sup"]["coord"].round(4).reset_index().rename(columns={"index":"variables"})
                return  DataTable(data = match_datalength(VarQuantCoord,input.VarQuantCoordLen()),
                                  filters=input.VarQuantCoordFilter())
            
            # Supplementary continuous variables cos2
            @output
            @render.data_frame
            def VarQuantCos2Table():
                VarQuantCos2 = get_mca_var(fa_model)["quanti_sup"]["cos2"].round(4).reset_index().rename(columns={"index":"variables"})
                return  DataTable(data = match_datalength(VarQuantCos2,input.VarQuantCos2Len()),
                                  filters=input.VarQuantCos2Filter())

            ###########################################################################################
            # Supplementary variables categories
            #-------------------------------------------------------------------------------------------
            ## Supplementary Variables/categories coordinates
            @output
            @render.data_frame
            def VarSupCoordTable():
                VarSupCoord = get_mca_mod(fa_model)["sup"]["coord"].round(4).reset_index().rename(columns={"index" : "Categories"})
                return DataTable(data=match_datalength(data=VarSupCoord,value=input.VarSupCoordLen()),
                                 filters=input.VarSupCoordFilter())
            
            #----------------------------------------------------------------------------------------
            # Supplementary variables/categories Cosinus
            @output
            @render.data_frame
            def VarSupCos2Table():
                VarSupCos2 = get_mca_mod(fa_model)["sup"]["cos2"].round(4).reset_index().rename(columns={"index" : "Categories"})
                return DataTable(data=match_datalength(data=VarSupCos2,value=input.VarSupCos2Len()),
                                 filters=input.VarSupCos2Filter())
            
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.VarSupCos2GraphBtn)
            def _():
                GraphModalShow(text="VarSup",name="Cos2")

            # Plot variables Cos2
            @output
            @render.plot(alt="Supplementary variables categories Cosines Map - MCA")
            def VarSupCos2Plot():
                VarSupCos2Fig = fviz_cosines(self=fa_model,
                                            choice="quali_sup",
                                            axis=input.VarSupCos2Axis(),
                                            top_cos2=int(input.VarSupCos2Top()),
                                            color=input.VarSupCos2Color(),
                                            bar_width=input.VarSupCos2BarWidth())
                return VarSupCos2Fig.draw()
            
            # Add Variables cosinus/correlations Modal show
            @reactive.Effect
            @reactive.event(input.VarSupCos2CorrGraphBtn)
            def _():
                GraphModelModal2(text="VarSup",name="Cos2",title=None)
            
            # Add reactive figure
            @reactive.Calc
            def VarSupCos2CorrMap():
                VarSupCos2 = get_mca_mod(fa_model)["sup"]["cos2"]
                fig = fviz_corrplot(
                    X=VarSupCos2,
                    title = input.VarSupCos2CorrTitle(),
                    outline_color=input.VarSupCos2CorrColor(),
                    colors=[input.VarSupCos2CorrLowColor(),
                            input.VarSupCos2CorrMidColor(),
                            input.VarSupCos2CorrHightColor()
                            ],
                        ylabel="Dimensions",
                        xlabel="Supplementary variables categories"
                    )+pn.theme_gray()
                return fig
            
            # Plot variables cosinus/correlations
            @output
            @render.plot(alt="Supplementary Variables categories Cosinus/correlations Map - MCA")
            def VarSupCos2CorrPlot():
                return VarSupCos2CorrMap().draw()
            
            ############################################################################################
            #    Tab : Description automatique des axes
            ############################################################################################
            #---------------------------------------------------------------------------------------
            # Description of axis
            @output
            @render.ui
            def DimDesc():
                if fa_model.quanti_sup_labels_ is not None:
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
                DimDesc = dimdesc(self=fa_model,axis=None,proba=input.pvalueDimdesc())
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
                DimDesc = dimdesc(self=fa_model,axis=None,proba=input.pvalueDimdesc())
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
            ### Statistiques descriptives
            @output
            @render.data_frame
            def StatsDescTable():
                StatsDesc = fa_model.original_data_.describe(include="all").round(4).T.reset_index().rename(columns={"index":"Variables"})
                return  DataTable(data = match_datalength(StatsDesc,input.StatsDescLen()),
                                filters=input.StatsDescFilter())

            #---------------------------------------------------------------------------------------------
            # Diagramme en barres
            @output
            @render.plot(alt="Bar-Plot")
            def VarBarPlotGraph():
                data = fa_model.original_data_
                if fa_model.quali_sup_labels_ is not None:
                    quali_sup = fa_model.data_[fa_model.quali_sup_labels_]
                    data = pd.concat([data,quali_sup],axis=1)
                    if fa_model.row_sup_labels_ is not None:
                        data = data.drop(index=fa_model.row_sup_labels_)
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
                overalldata = fa_model.data_.reset_index()
                return DataTable(data = match_datalength(overalldata,input.OverallDataLen()),
                                filters=input.OverallDataFilter())
            
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


