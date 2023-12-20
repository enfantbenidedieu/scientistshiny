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
    fviz_pca_ind,
    fviz_pca_var,
    fviz_eig, 
    fviz_contrib,
    fviz_cosines,
    fviz_corrplot)
from scientisttools.extractfactor import (
    get_eig,
    get_pca_var,
    get_pca_ind,
    dimdesc)

from scientistshiny.function import *

colors = mcolors.CSS4_COLORS
colors["cos2"] = "cos2"
colors["contrib"] = "contrib"

css_path = Path(__file__).parent / "www" / "style.css"

class PCAshiny:
    """
    Principal Component Analysis (PCA) with scientistshiny

    Description
    -----------
    Performs Principal Component Analysis (PCA) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables on a Shiny application.
    Graphics can be downloaded in png, jpg and pdf.

    Usage
    -----
    PCAshiny(fa_model)

    Parameters:
    ----------
    fa_model : An instance of class PCA. A PCA result from scientisttools.

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
        if fa_model.model_ != "pca":
            raise ValueError("Error : 'fa_model' must be an instance of class PCA")
        
        # -----------------------------------------------------------------------------------
        # Initialise value choice
        value_choice = {"EigenRes":"Valeurs propres",
                        "VarRes":"Résultats des variables",
                        "IndRes":"Résultats sur les individus"}
        if fa_model.row_sup_labels_ is not None:
            value_choice.update({"IndSupRes" : "Résultats des individus supplémentaires"})
        if fa_model.quanti_sup_labels_ is not None:
            value_choice.update({"VarSupRes" : "Résultats sur les variables supplémentaires"})
        if fa_model.quali_sup_labels_ is not None:
            value_choice.update({"VarQualRes" : "Résultats des variables qualitatives"})
        
        # Quantitatives columns
        col_labels = list(fa_model.col_labels_)
        if fa_model.quanti_sup_labels_ is not None:
            for i in range(len(fa_model.quanti_sup_labels_)):
                col = fa_model.quanti_sup_labels_[i]
                col_labels.append(col)

        
        DimDescChoice = {}
        for i in range(min(3,fa_model.n_components_)):
            DimDescChoice.update({"Dim."+str(i+1) : "Dimension "+str(i+1)})

        app_ui = ui.page_fluid(
            ui.include_css(css_path),
            shinyswatch.theme.superhero(),
            ui.page_navbar(title=ui.div(ui.panel_title(ui.h2("Analyse en Composantes Principales"),window_title="PCAshiny"),align="center"),inverse=True,id="navbar_id",padding={"style": "text-align: center;"}),
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
                            id="IndVar",
                            label="Modifier le graphe des",
                            choices={
                                "IndPlot":"individus",
                                "VarPlot":"variables"
                            },
                            selected="IndPlot",
                            inline=True,
                            width="100%"
                        ),
                        style="display: inline-block;"
                    ),
                    ui.panel_conditional("input.IndVar ==='IndPlot'",
                        ui.input_text(
                            id="IndTitle",
                            label="Titre du graphe",
                            value=None,
                            width="100%"
                        ),
                        ui.output_ui("choixindmod"),
                        ui.output_ui("pointlabel"),
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
                            ui.div(ui.input_slider(id="IndLimCos2", label = "Libellés pour un cos2 plus grand que",min = 0, max = 1,value=0,step=0.05),align="center")              
                        ),
                        ui.panel_conditional("input.IndPointSelect === 'contrib'",
                            ui.div(ui.input_slider(id="IndLimContrib", label ="Libellés pour une contribution plus grande que",min = 0, max = 100,value=0,step=5),align="center")              
                        ),
                        ui.input_select(
                            id="IndTextColor",
                            label="Colorier les points par :",
                            choices={
                                "actif/sup": "actifs/supplémentaires",
                                "cos2"     : "Cosinus",
                                "contrib"  : "Contribution",
                                "varquant" : "Variable quantitative",
                                "varqual"  : "Variable qualitative"
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
                            "input.IndTextColor==='varquant'",
                            ui.input_select(
                                id="IndTextVarQuant",
                                label="choix de la variable",
                                choices={x:x for x in col_labels},
                                selected=col_labels[0],
                                width="100%"
                            )
                        ),
                        ui.panel_conditional(
                            "input.IndTextColor==='varqual'",
                             ui.output_ui("IndTextVarQual"),
                        ),
                        ui.input_switch(
                            id="IndPlotRepel",
                            label="repel",
                            value=True
                        )
                    ),
                    ui.panel_conditional("input.IndVar ==='VarPlot'",
                        ui.input_text(
                                id="VarTitle",
                                label='Titre du graphe',
                                value=None,
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
                            id="VarSelect",
                            label="Libellés pour variables sélectionnées par",
                            choices={
                                "none"    : "Pas de sélection",
                                "cos2"    : "Cosinus",
                                "contrib" : "Contribution"
                            },
                            selected="none",
                            multiple=False,
                            width="100%"
                        ),
                        ui.panel_conditional("input.VarSelect === 'cos2'",
                            ui.div(
                                ui.input_slider(
                                    id="VarLimCos2",
                                    label = "Libellés pour un cos2 plus grand que",
                                    min = 0, 
                                    max = 1,
                                    value=0,
                                    step=0.05
                                ),
                                align="center"
                            )              
                        ),
                        ui.panel_conditional("input.VarSelect === 'contrib'",
                            ui.div(
                                ui.input_slider(
                                    id="VarLimContrib",
                                    label = "Libellés pour une contribution plus grande que",
                                    min = 0, 
                                    max = 100,
                                    value=0,
                                    step=5
                                ),
                                align="center"
                            )              
                        ),
                        ui.input_select(
                            id="VarTextColor",
                            label="Colorier les flèches par :",
                            choices={
                                "actif/sup" : "actives/supplémentaires",
                                "cos2"      : "Cosinus",
                                "contrib"   : "Contribution"
                            },
                            selected="actif/sup",
                            multiple=False,
                            width="100%"
                        ),
                        ui.panel_conditional(
                            "input.VarTextColor ==='actif/sup'",
                            ui.output_ui("VarTextChoice")
                        )
                    )
                    ),
                    ui.div(ui.input_action_button(id="exit",label="Quitter l'application",style='padding:5px; background-color: #fcac44;text-align:center;white-space: normal;'),align="center"),
                    width="25%"
                ),
                ui.navset_card_tab(
                    ui.nav("Graphes",
                        ui.row(
                            ui.column(7,
                                ui.div(ui.output_plot("RowFactorMap",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px"),
                                ui.div(ui.download_button(id="IndGraphDownloadJpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="IndGraphDownloadPng",label="png",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="IndGraphDownloadPdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                align="center"
                            ),
                            ui.column(5,
                                ui.div(ui.output_plot("VarFactorMap",width='100%', height='500px'),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                                ui.div(ui.download_button(id="var_download1",label="jpg",style = download_btn_style,icon=None),style="display: inline-block;",align="center"),
                                ui.div(ui.download_button(id="var_download2",label="png",style = "background-color: #1C2951;"),style="display: inline-block;",align="center"),
                                ui.div(ui.download_button(id="var_download3",label="pdf",style = "background-color: #1C2951;"),style="display: inline-block;",align="center"),
                                align="center"
                            )
                        )
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
                        OverallPanelConditional(text="Var"),
                        OverallPanelConditional(text="Ind"),
                        ui.output_ui("IndSupPanel"),
                        ui.output_ui("VarSupPanel"),  
                        ui.output_ui("VarQualPanel")
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
                            label="Que voulez - vous afficher?",
                            choices={
                                "StatsDesc":"Statistiques descriptives",
                                "Hist" : "Histogramme",
                                "CorrMatrix": "Matrice des corrélations"
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
                                    ui.input_select(id="VarLabel",label="Choisir une variable",choices={x:x for x in fa_model.active_data_.columns.values},selected=fa_model.active_data_.columns.values[0]),
                                    ui.input_switch(id="AddDensity",label="Densite",value=False)

                                ),
                                ui.column(10,ui.div(ui.output_plot(id="VarHistGraph",width='100%',height='500px'),align="center"))
                            )
                        ),
                        ui.panel_conditional("input.ResumeChoice==='CorrMatrix'",
                            PanelConditional1(text="CorrMatrix",name="")
                        )
                    ),
                    ui.nav("Données",
                        PanelConditional1(text="OverallData",name="")
                        
                    )
                )
            )
        )

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
                if fa_model.row_sup_labels_ is not None and fa_model.quali_sup_labels_ is not None:
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
                        ),
                        ui.input_select(
                            id="IndTextModColor",
                            label="modalités",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="red",
                            multiple=False,
                            width="100%"
                        ),
                    ),
                elif fa_model.row_sup_labels_ is not None:
                    return ui.TagList(
                        ui.input_select(
                            id="IndTextActifColor",
                            label="individus actifs",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="blue",
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
                elif fa_model.quali_sup_labels_ is not None:
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
                            id="IndTextModColor",
                            label="modalités",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="red",
                            multiple=False,
                            width="100%"
                        ),
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
            
            #-------------------------------------------------------------------------------------------------
            @output
            @render.ui
            def IndTextVarQual():
                if fa_model.quali_sup_labels_ is not None:
                    return ui.TagList(
                        ui.input_select(
                            id="IndTextVarQualColor",
                            label="Choix de la variable",
                            choices={x:x for x in fa_model.quali_sup_labels_},
                            selected=fa_model.quali_sup_labels_[0],
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_switch(
                            id="AddEllipse",
                            label="Trace les ellipses de confiance autour des modalités",
                            value=False
                        )
                    )
                else:
                    return ui.TagList(
                        ui.p("Aucune variable qualitative")
                    )
            
            #-----------------------------------------------------------------------------------------------
            @output
            @render.ui
            def VarTextChoice():
                if fa_model.quanti_sup_labels_ is not None:
                    return ui.TagList(
                        ui.input_select(
                            id="VarTextActifColor",
                            label="Variables actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="VarTextSupColor",
                            label="Variables supplémentaires",
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
                            label="Variables actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        )
                    )
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
            
            # Add Continuous Supplementary Conditional Panel
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
            
            # Add Categories Supplementary Conditional Panel
            @output
            @render.ui
            def VarQualPanel():
                return ui.panel_conditional("input.choice == 'VarQualRes'",
                            ui.br(),
                            ui.h5("Coordonnées"),
                            PanelConditional1(text="VarQual",name="Coord"),
                            ui.hr(),
                            ui.h5("V-test"),
                            PanelConditional2(text="VarQual",name="Vtest")
                        )

            # -------------------------------------------------------
            @reactive.Effect
            @reactive.event(input.exit)
            async def _():
                await session.close()

            #--------------------------------------------------------------------------------
            @reactive.Calc
            def RowPlot():
                if input.IndTextColor() == "actif/sup":
                    if (fa_model.row_sup_labels_ is not None) and (fa_model.quali_sup_labels_ is not None):
                        fig = fviz_pca_ind(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            color=input.IndTextActifColor(),
                            color_sup = input.IndTextSupColor(),
                            color_quali_sup=input.IndTextModColor(),
                            text_size = input.IndTextSize(),
                            lim_contrib =input.IndLimContrib(),
                            lim_cos2 = input.IndLimCos2(),
                            title = input.IndTitle(),
                            repel=input.IndPlotRepel()
                        )
                    elif fa_model.row_sup_labels_ is not None:
                        fig = fviz_pca_ind(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            color=input.IndTextActifColor(),
                            color_sup = input.IndTextSupColor(),
                            color_quali_sup=None,
                            text_size = input.IndTextSize(),
                            lim_contrib =input.IndLimContrib(),
                            lim_cos2 = input.IndLimCos2(),
                            title = input.IndTitle(),
                            repel=input.IndPlotRepel()
                        )
                    elif fa_model.quali_sup_labels_ is not None:
                        fig = fviz_pca_ind(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            color=input.IndTextActifColor(),
                            color_sup = None,
                            color_quali_sup=input.IndTextModColor(),
                            text_size = input.IndTextSize(),
                            lim_contrib =input.IndLimContrib(),
                            lim_cos2 = input.IndLimCos2(),
                            title = input.IndTitle(),
                            repel=input.IndPlotRepel()
                        )
                    else:
                        fig = fviz_pca_ind(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            color=input.IndTextActifColor(),
                            color_sup = None,
                            color_quali_sup=None,
                            text_size = input.IndTextSize(),
                            lim_contrib =input.IndLimContrib(),
                            lim_cos2 = input.IndLimCos2(),
                            title = input.IndTitle(),
                            repel=input.IndPlotRepel()
                        )
                elif input.IndTextColor() in ["cos2","contrib"]:
                    fig = fviz_pca_ind(
                        self=fa_model,
                        axis=[int(input.Axis1()),int(input.Axis2())],
                        color=input.IndTextColor(),
                        text_size = input.IndTextSize(),
                        lim_contrib =input.IndLimContrib(),
                        lim_cos2 = input.IndLimCos2(),
                        title = input.IndTitle(),
                        repel=input.IndPlotRepel()
                    )
                elif input.IndTextColor() == "varquant":
                    fig = fviz_pca_ind(
                        self=fa_model,
                        axis=[int(input.Axis1()),int(input.Axis2())],
                        color=input.IndTextVarQuant(),
                        text_size = input.IndTextSize(),
                        lim_contrib =input.IndLimContrib(),
                        lim_cos2 = input.IndLimCos2(),
                        title = input.IndTitle(),
                        legend_title=input.IndTextVarQuant(),
                        repel=input.IndPlotRepel()
                    )
                elif input.IndTextColor() == "varqual":
                    if fa_model.quali_sup_labels_ is not None:
                        fig = fviz_pca_ind(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            text_size = input.IndTextSize(),
                            lim_contrib =input.IndLimContrib(),
                            lim_cos2 = input.IndLimCos2(),
                            title = input.IndTitle(),
                            habillage= input.IndTextVarQualColor(),
                            add_ellipse=input.AddEllipse(),
                            repel=input.IndPlotRepel()
                        )
                    else:
                        fig = pn.ggplot()
                return fig

            # ------------------------------------------------------------------------------
            # Individual Factor Map - PCA
            @output
            @render.plot(alt="Individuals Factor Map - PCA")
            def RowFactorMap():
                return RowPlot().draw()
            
            # Downlaod
            #@session.download(filename="Individuals-Factor-Map.png")
            #def IndGraphDownloadPng():
            #    return pw.load_ggplot(RowPlot()).savefig("Individuals-Factor-Map.png")
            
            #############################################################################################
            #  Variables Factor Map
            ##############################################################################################

            @reactive.Calc
            def VarFactorPlot():
                if input.VarTextColor() == "actif/sup":
                    if fa_model.quanti_sup_labels_ is not None:
                        fig = fviz_pca_var(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            title=input.VarTitle(),
                            color=input.VarTextActifColor(),
                            color_sup=input.VarTextSupColor(),
                            text_size=input.VarTextSize(),
                            lim_contrib = input.VarLimContrib(),
                            lim_cos2 = input.VarLimCos2() 
                            )
                    else:
                        fig = fviz_pca_var(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            title=input.VarTitle(),
                            color=input.VarTextActifColor(),
                            color_sup=None,
                            text_size=input.VarTextSize(),
                            lim_contrib = input.VarLimContrib(),
                            lim_cos2 = input.VarLimCos2() 
                            )
                elif input.VarTextColor() in ["cos2","contrib"]:
                    if fa_model.quanti_sup_labels_ is not None:
                        fig = fviz_pca_var(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            title=input.VarTitle(),
                            color=input.VarTextColor(),
                            color_sup=input.VarTextSupColor(),
                            text_size=input.VarTextSize(),
                            lim_contrib = input.VarLimContrib(),
                            lim_cos2 = input.VarLimCos2() 
                            )
                    else:
                        fig = fviz_pca_var(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            title=input.VarTitle(),
                            color=input.VarTextColor(),
                            color_sup=None,
                            text_size=input.VarTextSize(),
                            lim_contrib = input.VarLimContrib(),
                            lim_cos2 = input.VarLimCos2() 
                            )
                return fig

            # Variables Factor Map - PCA
            @output
            @render.plot(alt="Variables Factor Map - PCA")
            def VarFactorMap():
                return VarFactorPlot().draw()
            
            #-------------------------------------------------------------------------------------------
            # Eigenvalue - Scree plot
            @output
            @render.plot(alt="Scree Plot - PCA")
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
            
            ##################################################################################################
            #   Continuous Variables
            #####################################################################################################
            #---------------------------------------------------------------------------------------------
            # Variables Coordinates
            @output
            @render.data_frame
            def VarCoordTable():
                VarCoord = get_pca_var(fa_model)["coord"].round(4).reset_index().rename(columns={"index" : "Variables"})
                return DataTable(data=match_datalength(data=VarCoord,value=input.VarCoordLen()),
                                filters=input.VarCoordFilter())
            
            #----------------------------------------------------------------------------------------------------
            # Variables Contributions
            @output
            @render.data_frame
            def VarContribTable():
                VarContrib = get_pca_var(fa_model)["contrib"].round(4).reset_index().rename(columns={"index" : "Variables"})
                return  DataTable(data=match_datalength(data=VarContrib,value=input.VarContribLen()),
                                filters=input.VarContribFilter())
            
            #-----------------------------------------------------------------------------------------------------
            # Add Variables Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.VarContribGraphBtn)
            def _():
                GraphModalShow(text="Var",name="Contrib")
            
            @reactive.Calc
            def VarContribMap():
                fig = fviz_contrib(
                    self=fa_model,
                    choice="var",
                    axis=input.VarContribAxis(),
                    top_contrib=int(input.VarContribTop()),
                    color=input.VarContribColor(),
                    bar_width=input.VarContribBarWidth()
                    )
                return fig

            # Plot variables Contributions
            @output
            @render.plot(alt="Variables Contributions Map - PCA")
            def VarContribPlot():
                return VarContribMap().draw()
            
            #----------------------------------------------------------------------------------------------------------------
            # Add Variables Contributions Correlation Modal Show
            @reactive.Effect
            @reactive.event(input.VarContribCorrGraphBtn)
            def _():
                GraphModelModal2(text="Var",name="Contrib",title=None)
            
            @reactive.Calc
            def VarContribCorrMap():
                VarContrib = get_pca_var(fa_model)["contrib"]
                fig = fviz_corrplot(
                    X=VarContrib,
                    title=input.VarContribCorrTitle(),
                    outline_color=input.VarContribCorrColor(),
                    colors=[input.VarContribCorrLowColor(),
                            input.VarContribCorrMidColor(),
                            input.VarContribCorrHightColor()
                            ]
                    )+pn.theme_gray()
                return fig

            # Plot variables Contributions/correlations Map - PCA
            @output
            @render.plot(alt="Variables Contributions/Correlations Map - PCA")
            def VarContribCorrPlot():
                return VarContribCorrMap().draw()
            
            #-----------------------------------------------------------------------------------------------------------
            # Variables Cos2 
            @output
            @render.data_frame
            def VarCos2Table():
                VarCos2 = get_pca_var(fa_model)["cos2"].round(4).reset_index().rename(columns={"index" : "Variables"})
                return  DataTable(data=match_datalength(data=VarCos2,value=input.VarCos2Len()),
                                filters=input.VarCos2Filter())
            
            #-------------------------------------------------------------------------------------------------------------
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.VarCos2GraphBtn)
            def _():
                GraphModalShow(text="Var",name="Cos2")
            
            @reactive.Calc
            def VarCos2Map():
                fig = fviz_cosines(
                    self=fa_model,
                    choice="var",
                    axis=input.VarCos2Axis(),
                    top_cos2=int(input.VarCos2Top()),
                    color=input.VarCos2Color(),
                    bar_width=input.VarCos2BarWidth())
                return fig

            # Plot variables Cos2
            @output
            @render.plot(alt="Variables Cosines Map - PCA")
            def VarCos2Plot():
                return VarCos2Map().draw()
            
            #----------------------------------------------------------------------------------------
            # Add Variables Cosinus Correlation Modal Show
            @reactive.Effect
            @reactive.event(input.VarCos2CorrGraphBtn)
            def _():
                GraphModelModal2(text="Var",name="Cos2",title=None)
            
            @reactive.Calc
            def VarCos2CorrMap():
                VarCos2 = get_pca_var(fa_model)["cos2"]
                fig = fviz_corrplot(
                    X=VarCos2,
                    title=input.VarCos2CorrTitle(),
                    outline_color=input.VarCos2CorrColor(),
                    colors=[input.VarCos2CorrLowColor(),
                            input.VarCos2CorrMidColor(),
                            input.VarCos2CorrHightColor()
                            ])+pn.theme_gray()
                return fig

            #--------------------------------------------------------------------------------------------------
            # Plot variables Contributions
            @output
            @render.plot(alt="Variables Contributions/Correlations Map - PCA")
            def VarCos2CorrPlot():
                return VarCos2CorrMap().draw()

            #---------------------------------------------------------------------------------
            ## Supplementary Continuous Variables
            # Continuous Variables Coordinates
            @output
            @render.data_frame
            def VarSupCoordTable():
                VarSupCoord = get_pca_var(fa_model)["quanti_sup"]["coord"].round(4).reset_index().rename(columns={"index" : "Variables"})
                return DataTable(data=match_datalength(data=VarSupCoord,value=input.VarSupCoordLen()),
                                filters=input.VarSupCoordFilter())
            
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.VarSupCorrGraphBtn)
            def _():
                GraphModalShow(text="VarSup",name="Coord")

            # Plot variables Cos2
            @output
            @render.plot(alt="Supplementary Continuous Variables Coordinates Map - PCA")
            def VarSupCoordPlot():
                VarSupCoord = get_pca_var(fa_model)["quanti_sup"]["coord"]
                VarSupCoordFig = fviz_barplot(X=VarSupCoord,
                                            ncp=fa_model.n_components_,
                                            axis=input.VarSupCoordAxis(),
                                            top_corr=int(input.VarSupCoordTop()),
                                            color=input.VarSupCoordColor(),
                                            bar_width=input.VarSupCoordBarWidth(),
                                            xlabel="Correlation",
                                            ylabel="Variables",
                                            title=f"Correlation of supplementary continuous variables to Dim-{input.VarSupCoordAxis()+1}")
                return VarSupCoordFig.draw()
            
            #----------------------------------------------------------------------------------------
            # Supplementary Cosinus
            @output
            @render.data_frame
            def VarSupCos2Table():
                VarSupCos2 = get_pca_var(fa_model)["quanti_sup"]["cos2"].round(4).reset_index().rename(columns={"index" : "Variables"})
                return DataTable(data=match_datalength(data=VarSupCos2,value=input.VarSupCos2Len()),
                                filters=input.VarSupCos2Filter())
            
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.VarSupCos2GraphBtn)
            def _():
                GraphModalShow(text="VarSup",name="Cos2")

            # Plot variables Cos2
            @output
            @render.plot(alt="Supplementary continues variables Cosines Map - PCA")
            def VarSupCos2Plot():
                VarSupCos2Fig = fviz_cosines(self=fa_model,
                                          choice="quanti_sup",
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
                VarSupCos2 = get_pca_var(fa_model)["quanti_sup"]["cos2"]
                fig = fviz_corrplot(
                    X=VarSupCos2,
                    title = input.VarSupCos2CorrTitle(),
                    outline_color=input.VarSupCos2CorrColor(),
                    colors=[input.VarSupCos2CorrLowColor(),
                            input.VarSupCos2CorrMidColor(),
                            input.VarSupCos2CorrHightColor()
                            ],
                        ylabel="actives Variables",
                        xlabel="Supplementary continuous variables"
                    )+pn.theme_gray()
                return fig
            
            # Plot variables cosinus/correlations
            @output
            @render.plot(alt="Supplementary Continuous Variables Cosinus/correlations Map - PCA")
            def VarSupCos2CorrPlot():
                return VarSupCos2CorrMap().draw()
            
            ########################################################################################################
            # Individuals informations
            ##########################################################################################################
            #---------------------------------------------------------------------------------------------
            # Individuals Coordinates
            @output
            @render.data_frame
            def IndCoordTable():
                IndCoord = get_pca_ind(fa_model)["coord"].round(4).reset_index()
                return DataTable(data = match_datalength(IndCoord,input.IndCoordLen()),
                                filters=input.IndCoordFilter())
            
            # Individuals Contributions
            @output
            @render.data_frame
            def IndContribTable():
                IndContrib = get_pca_ind(fa_model)["contrib"].round(4).reset_index()
                return  DataTable(data=match_datalength(IndContrib,input.IndContribLen()),
                                filters=input.IndContribFilter())
            
            # Add indiviuals Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.IndContribGraphBtn)
            def _():
                GraphModalShow(text="Ind",name="Contrib")

            # Plot Individuals Contributions
            @output
            @render.plot(alt="Individuals Contributions Map - PCA")
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
            @render.plot(alt="Individuals Contributions/Correlations Map - PCA")
            def IndContribCorrPlot():
                IndContrib = get_pca_ind(fa_model)["contrib"]
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
                IndCos2 = get_pca_ind(fa_model)["cos2"].round(4).reset_index()
                return  DataTable(data = match_datalength(IndCos2,input.IndCos2Len()),
                                filters=input.IndCos2Filter())
            
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.IndCos2GraphBtn)
            def _():
                GraphModalShow(text="Ind",name="Cos2")

            # Plot variables Cos2
            @output
            @render.plot(alt="Individuals Cosines Map - PCA")
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
            @render.plot(alt="Individuals Cosinus/Correlations Map - PCA")
            def IndCos2CorrPlot():
                IndCos2 = get_pca_ind(fa_model)["cos2"]
                IndCos2CorrFig = fviz_corrplot(X=IndCos2,
                                            title=input.IndCos2CorrTitle(),
                                            outline_color=input.IndCos2CorrColor(),
                                            colors=[input.IndCos2CorrLowColor(),
                                                    input.IndCos2CorrMidColor(),
                                                    input.IndCos2CorrHightColor()
                                                ])+pn.theme_gray()
                return IndCos2CorrFig.draw()
            
            #----------------------------------------------------------------------------
            # Supplementaru Individual Coordinates
            @output
            @render.data_frame
            def IndSupCoordTable():
                IndSupCoord = get_pca_ind(fa_model)["ind_sup"]["coord"].round(4).reset_index()
                return  DataTable(data = match_datalength(IndSupCoord,input.IndSupCoordLen()),
                                filters=input.IndSupCoordFilter())
            
            # Supplementaru Individual Cos2
            @output
            @render.data_frame
            def IndSupCos2Table():
                IndSupCos2 = get_pca_ind(fa_model)["ind_sup"]["cos2"].round(4).reset_index()
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
                IndSupCos2 = get_pca_ind(fa_model)["ind_sup"]["cos2"]
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
                IndSupCos2 = get_pca_ind(fa_model)["ind_sup"]["cos2"]
                IndSupCos2CorrFig = fviz_corrplot(X=IndSupCos2,
                                                title=input.IndSupCos2CorrTitle(),
                                                outline_color=input.IndSupCos2CorrColor(),
                                                colors=[input.IndSupCos2CorrLowColor(),
                                                        input.IndSupCos2CorrMidColor(),
                                                        input.IndSupCos2CorrHightColor()
                                                        ],
                                                    xlabel="Supplementary individuals")+pn.theme_gray()
                return IndSupCos2CorrFig.draw()

            #-------------------------------------------------------------------------------------------
            # Supplementary categories variables
            @output
            @render.data_frame
            def VarQualCoordTable():
                VarQualCoord = get_pca_var(fa_model)["quali_sup"]["coord"].round(4).reset_index()
                return  DataTable(data = match_datalength(VarQualCoord,input.VarQualCoordLen()),
                                filters=input.VarQualCoordFilter())
            
            # Value - Test categories variables
            @output
            @render.data_frame
            def VarQualVtestTable():
                VarQualVtest = get_pca_var(fa_model)["quali_sup"]["vtest"].round(4).reset_index()
                return  DataTable(data = match_datalength(VarQualVtest,input.VarQualVtestLen()),
                                filters=input.VarQualVtestFilter())
            
            # Add Variables Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.VarQualVtestGraphBtn)
            def _():
                GraphModalShow(text="VarQual",name="Vtest")
            
            # Plot categories V-test
            @output
            @render.plot(alt="Supplementary categories V-test barplot Map - PCA")
            def VarQualVtestPlot():
                VarQualVtest = get_pca_var(fa_model)["quali_sup"]["vtest"]
                VarQualVtestFig = fviz_barplot(X=VarQualVtest,
                                            axis=input.VarQualVtestAxis(),
                                            top_corr=int(input.VarQualVtestTop()),
                                            color=input.VarQualVtestColor(),
                                            bar_width=input.VarQualVtestBarWidth(),
                                            ylabel="Supplementary categories",
                                            xlabel="V-test",
                                            title=f"V-test of supplementary categories to Dim-{input.VarQualVtestAxis()+1}")
                return VarQualVtestFig.draw()
            
            # Add Variables Cosines Correlation Modal Show
            @reactive.Effect
            @reactive.event(input.VarQualVtestCorrGraphBtn)
            def _():
                GraphModelModal2(text="VarQual",name="Vtest",title=None)

            # Plot variables Contributions
            @output
            @render.plot(alt="Supplementary Categories Vtest Map - PCA")
            def VarQualVtestCorrPlot():
                VarQualVtest = get_pca_var(fa_model)["quali_sup"]["vtest"]
                VarQualVtestCorrFig = fviz_corrplot(X=VarQualVtest,
                                                title=input.VarQualVtestCorrTitle(),
                                                outline_color=input.VarQualVtestCorrColor(),
                                                colors=[input.VarQualVtestCorrLowColor(),
                                                        input.VarQualVtestCorrMidColor(),
                                                        input.VarQualVtestCorrHightColor()
                                                        ],
                                                    xlabel="Supplementary categories")+pn.theme_gray()
                return VarQualVtestCorrFig.draw()
            
            #---------------------------------------------------------------------------------------
            # Description of axis
            @output
            @render.ui
            def DimDesc():
                if fa_model.quali_sup_labels_ is not None:
                    return ui.TagList(
                        ui.h5("Quantitative"),
                        PanelConditional1(text="Dim1",name="Desc"),
                        ui.hr(),
                        ui.h5("Qualitative"),
                        PanelConditional1(text="Dim2",name="Desc"),
                    )
                else:
                    return ui.TagList(
                        ui.h5("Quantitative"),
                        PanelConditional1(text="Dim1",name="Desc")
                    )
            
            @output
            @render.data_frame
            def Dim1DescTable():
                DimDesc = dimdesc(self=fa_model,axis=None,proba=input.pvalueDimdesc())
                if isinstance(DimDesc[input.Dimdesc()],dict):
                    DimDescQuanti = DimDesc[input.Dimdesc()]["quanti"].reset_index().rename(columns={"index":"Variables"})
                elif isinstance(DimDesc[input.Dimdesc()],pd.DataFrame):
                    DimDescQuanti = DimDesc[input.Dimdesc()].reset_index().rename(columns={"index":"Variables"})
                else:
                    DimDescQuanti = pd.DataFrame()
                return  DataTable(data = match_datalength(DimDescQuanti,input.Dim1DescLen()),
                                filters=input.Dim1DescFilter())
            
            @output
            @render.data_frame
            def Dim2DescTable():
                DimDesc = dimdesc(self=fa_model,axis=None,proba=input.pvalueDimdesc())
                if isinstance(DimDesc[input.Dimdesc()],dict):
                    DimDescQuali = DimDesc[input.Dimdesc()]["quali"].reset_index().rename(columns={"index":"Variables"})
                else:
                    DimDescQuali = pd.DataFrame()
                return  DataTable(data = match_datalength(DimDescQuali,input.Dim2DescLen()),
                                filters=input.Dim2DescFilter())
            
            #-----------------------------------------------------------------------------------------------
            ### Statistiques descriptives
            @output
            @render.data_frame
            def StatsDescTable():
                StatsDesc = fa_model.active_data_.describe(include="all").round(4).T.reset_index().rename(columns={"index":"Variables"})
                return  DataTable(data = match_datalength(StatsDesc,input.StatsDescLen()),
                                filters=input.StatsDescFilter())

            # Histogramme
            @output
            @render.plot(alt="")
            def VarHistGraph():
                p = pn.ggplot(fa_model.active_data_,pn.aes(x=input.VarLabel()))
                # Add density
                if input.AddDensity():
                    p = (p + pn.geom_histogram(pn.aes(y="..density.."), color="darkblue", fill="lightblue")+
                        pn.geom_density(alpha=.2, fill="#FF6666"))
                else:
                    p = p + pn.geom_histogram(color="darkblue", fill="lightblue")
                
                p = p + pn.ggtitle(f"Histogram de {input.VarLabel()}")

                return p.draw()

            # Matrice des corrélations
            @output
            @render.data_frame
            def CorrMatrixTable():
                corr_mat = fa_model.active_data_.corr(method="pearson").round(4).reset_index().rename(columns={"index":"Variables"})
                return DataTable(data = match_datalength(corr_mat,input.CorrMatrixLen()),
                                filters=input.CorrMatrixFilter())
            
            ##############################################################################################
            # Overall Daat
            ###############################################################################################################
            # Overall Data
            @output
            @render.data_frame
            def OverallDataTable():
                overalldata = fa_model.data_.reset_index()
                return DataTable(data = match_datalength(overalldata,input.OverallDataLen()),
                                filters=input.OverallDataFilter())
            
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
    
    # Stop App
    def stop(self):
        """
        
        
        """
        app = App(ui=self.app_ui, server=self.server)
        return app.stop()


