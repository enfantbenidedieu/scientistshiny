# -*- coding: utf-8 -*-
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
import shinyswatch
from pathlib import Path
import numpy as np
import pandas as pd
#import patchworklib as pw
import plotnine as pn
import matplotlib.colors as mcolors
import nest_asyncio
import uvicorn

from sklearn.base import BaseEstimator, TransformerMixin
from scientisttools import fviz_ca_row,fviz_ca_col,fviz_eig, fviz_contrib,fviz_cos2,fviz_corrplot,dimdesc

from .function import *

colors = mcolors.CSS4_COLORS
colors["cos2"] = "cos2"
colors["contrib"] = "contrib"

css_path = Path(__file__).parent / "www" / "style.css"

class CAshiny(BaseEstimator,TransformerMixin):
    """
    Correspondance Analysis (CA) with scientistshiny
    ------------------------------------------------

    Description
    -----------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs Correspondance Analysis (PCA) including supplementary row and/or column points on a Shiny application.
    Graphics can be downloaded in png, jpg and pdf.

    Usage
    -----
    CAshiny(model)

    Parameters:
    ----------
    model : An object of class CA. A CA result from scientisttools.

    Returns:
    -------
    Graphs : a tab containing the the row and column points factor map (with supplementary columns and supplementary rows)

    Values : a tab containing the summary of the CA performed, the eigenvalues, the results
             for the columns, for the rows, for the supplementary columns and for the supplementarry rows
             variables and the results for the categorical variables.

    Automatic description of axes : a tab containing the output of the dimdesc function. This function is designed to 
                                    point out the variables and the categories that are the most characteristic according
                                    to each dimension obtained by a Factor Analysis.

    Summary of dataset : A tab containing the summary of the dataset and a boxplot and histogramm for quantitative variables.

    Data : a tab containing the dataset with a nice display.

    The left part of the application allows to change some elements of the graphs (axes, variables, colors,.)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples:
    ---------
    > from scientisttools import CA
    > from scientistshiny import CAshiny


    for jupyter notebooks
    https://stackoverflow.com/questions/74070505/how-to-run-fastapi-application-inside-jupyter
    """
    def __init__(self,model=None):
        # Check if model is Correspondence Analysis (CA)
        if model.model_ != "ca":
            raise TypeError("'model' must be an instance of class CA")
        
        # -----------------------------------------------------------------------------------
        # Initialise value choice
        value_choice = {"EigenRes": "Valeurs propres",
                        "ColRes"  : "Résultats pour les colonnes",
                        "RowRes"  : "Résultats pour les lignes"}
        # Check if supplementary rows
        if hasattr(model,"row_sup_"):
            value_choice.update({"RowSupRes" : "Résultats pour les lignes supplémentaires"})
        
        # Check if supplementary columns
        if hasattr(model,"col_sup_"):
            value_choice.update({"ColSupRes" : "Résultats pour les colonnes supplémentaires"})
        
        # Check if supplementary quantitatives columns
        if hasattr(model,"quanti_sup_"):
            value_choice.update({"QuantiSupRes" : "Résultats pour les variables quantitatives supplémentaires"})

        # Check if supplementary qualitatives columns
        if hasattr(model,"quali_sup_"):
            value_choice.update({"QualiSupRes" : "Résultats pour les variables qualitatives supplémentaires"})

        # Dimension to return
        nbDim = min(3,model.call_["n_components"])
        DimDescChoice = {}
        for i in range(nbDim):
            DimDescChoice.update({"Dim."+str(i+1) : "Dimension "+str(i+1)})

        # App UI
        app_ui = ui.page_fluid(
            ui.include_css(css_path),
            shinyswatch.theme.superhero(),
            ui.page_navbar(title=ui.div(ui.panel_title(ui.h2("Analyse Factorielle des Correspondances"),window_title="CAshiny"),align="center"),inverse=True,id="navbar_id",padding={"style": "text-align: center;"}),
            ui.page_sidebar(
                ui.sidebar(
                    ui.panel_well(
                        ui.h6("Options graphiques",style="text-align : center"),
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
                            id="RowCol",
                            label="Modifier le graphe des",
                            choices={
                                "RowPlot":"Points lignes",
                                "ColPlot":"Points colonnes"
                            },
                            selected="RowPlot",
                            inline=True,
                            width="100%"
                        ),
                        style="display: inline-block;"
                    ),
                    ui.panel_conditional("input.RowCol ==='RowPlot'",
                        ui.input_text(
                            id="RowTitle",
                            label="Titre du graphe",
                            value="Row points - CA",
                            width="100%"
                        ),
                        ui.output_ui("choixindmod"),
                        ui.output_ui("pointlabel"),
                        ui.input_slider(
                            id="RowTextSize",
                            label="Taille des libellés",
                            min=8,
                            max=20,
                            value=8,
                            step=2,
                            ticks=False
                        ),
                        ui.input_select(
                            id="RowPointSelect",
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
                        ui.panel_conditional("input.RowPointSelect === 'cos2'",
                            ui.div(ui.input_slider(id="RowLimCos2", label = "Libellés pour un cos2 plus grand que",min = 0, max = 1,value=0,step=0.05),align="center")              
                        ),
                        ui.panel_conditional("input.RowPointSelect === 'contrib'",
                            ui.div(ui.input_slider(id="RowLimContrib", label ="Libellés pour une contribution plus grande que",min = 0, max = 100,value=0,step=5),align="center")              
                        ),
                        ui.input_select(
                            id="RowTextColor",
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
                            "input.RowTextColor==='actif/sup'",
                            ui.output_ui("RowTextChoice"),
                        ),
                        ui.input_switch(
                            id="RowPlotRepel",
                            label="repel",
                            value=True
                        )
                    ),
                    ui.panel_conditional("input.RowCol ==='ColPlot'",
                        ui.input_text(
                                id="ColTitle",
                                label='Titre du graphe',
                                value="Columns points - CA",
                                width="100%"
                        ),
                        ui.input_slider(
                            id="ColTextSize",
                            label="Taille des libellés",
                            min=8,
                            max=20,
                            value=8,
                            step=2,
                            ticks=False
                        ),
                        ui.input_select(
                            id="ColSelect",
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
                        ui.panel_conditional("input.ColSelect === 'cos2'",
                            ui.div(
                                ui.input_slider(
                                    id="ColLimCos2",
                                    label = "Libellés pour un cos2 plus grand que",
                                    min = 0, 
                                    max = 1,
                                    value=0,
                                    step=0.05
                                ),
                                align="center"
                            )              
                        ),
                        ui.panel_conditional("input.ColSelect === 'contrib'",
                            ui.div(
                                ui.input_slider(
                                    id="ColLimContrib",
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
                            id="ColTextColor",
                            label="Colorier les flèches par :",
                            choices={
                                "actif/sup" : "actif/supplémentaire",
                                "cos2"      : "Cosinus",
                                "contrib"   : "Contribution"
                            },
                            selected="actif/sup",
                            multiple=False,
                            width="100%"
                        ),
                        ui.panel_conditional(
                            "input.ColTextColor ==='actif/sup'",
                            ui.output_ui("ColTextChoice")
                        ),
                        ui.input_switch(
                            id="ColPlotRepel",
                            label="repel",
                            value=True
                        )
                    )
                    ),
                    ui.div(ui.input_action_button(id="exit",label="Quitter l'application",style='padding:5px; background-color: #fcac44;text-align:center;white-space: normal;'),align="center"),
                    width="25%"
                ),
                ui.navset_card_tab(
                    ui.nav_panel("Graphes",
                        ui.row(
                            ui.column(6,
                                ui.div(ui.output_plot("RowFactorMap",width='100%',height="600px",fill=True),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px"),
                                ui.div(ui.download_button(id="RowGraphDownloadJpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="RowGraphDownloadPng",label="png",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="RowGraphDownloadPdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                align="center"
                            ),
                            ui.column(6,
                                ui.div(ui.output_plot("ColFactorMap",width='100%',height="600px"),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                                ui.div(ui.download_button(id="ColGraphDownloadJpg",label="jpg",style = download_btn_style,icon=None),style="display: inline-block;",align="center"),
                                ui.div(ui.download_button(id="ColGraphDownloadPng",label="png",style = "background-color: #1C2951;"),style="display: inline-block;",align="center"),
                                ui.div(ui.download_button(id="ColGraphDownloadPdf",label="pdf",style = "background-color: #1C2951;"),style="display: inline-block;",align="center"),
                                align="center"
                            )
                        )
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
                        OverallPanelConditional(text="Col"),
                        OverallPanelConditional(text="Row"),
                        ui.output_ui("RowSupPanel"),
                        ui.output_ui("ColSupPanel"),
                        ui.output_ui("QuantiSupPanel"),
                        ui.output_ui("QualiSupPanel")
                    ),
                    ui.nav_panel("Description automatique des axes",
                        ui.input_radio_buttons(id="Dimdesc",
                                               label="Choisir les dimensions",
                                               choices=DimDescChoice,selected="Dim.1",inline=True),
                        ui.output_ui(id="DimDesc")
                    ),
                    ui.nav_panel("Résumé du jeu de données",
                        ui.h5("Distributions conditionelles (X/Y)"),
                        PanelConditional1(text="CondDist",name="One"),
                        ui.hr(),
                        ui.h5("Distributions conditionnelles (Y/X)"),
                        PanelConditional1(text="CondDist",name="Two") 
                    ),
                    ui.nav_panel("Données",
                                 PanelConditional1(text="OverallData",name=""))
                )
            )
        )

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
            def RowTextChoice():
                if model.row_sup is not None:
                    return ui.TagList(
                        ui.input_select(
                            id="RowTextActifColor",
                            label="lignes actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="RowTextSupColor",
                            label="lignes supplémentaires",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="blue",
                            multiple=False,
                            width="100%"
                        )
                    )
                else:
                    return ui.TagList(
                        ui.input_select(
                            id="RowTextActifColor",
                            label="lignes actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        )
                    )
            #-----------------------------------------------------------------------------------------------
            @output
            @render.ui
            def ColTextChoice():
                if model.col_sup is not None:
                    return ui.TagList(
                        ui.input_select(
                            id="ColTextActifColor",
                            label="colonnes actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        ),
                        ui.input_select(
                            id="ColTextSupColor",
                            label="colonnes supplémentaires",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="red",
                            multiple=False,
                            width="100%"
                        )
                    )
                else:
                    return ui.TagList(
                        ui.input_select(
                            id="ColTextActifColor",
                            label="colonnes actives",
                            choices={x:x for x in mcolors.CSS4_COLORS},
                            selected="black",
                            multiple=False,
                            width="100%"
                        )
                    )
            #-------------------------------------------------------------------------------------------------
            # Add Supplementary Rows Conditional Panel
            @output
            @render.ui
            def RowSupPanel():
                return ui.panel_conditional("input.choice == 'RowSupRes'",
                            ui.br(),
                            ui.h5("Coordonnées"),
                            PanelConditional1(text="RowSup",name="Coord"),
                            ui.hr(),
                            ui.h5("Cos2 - Qualité de la représentation"),
                            PanelConditional1(text="RowSup",name="Cos2") 
                        )
            
            
            
            # -------------------------------------------------------
            @reactive.Effect
            @reactive.event(input.exit)
            async def _():
                await session.close()

            #--------------------------------------------------------------------------------
            @reactive.Calc
            def RowFactorPlot():
                if input.RowTextColor() == "actif/sup":
                    if model.row_sup is None:
                        fig = fviz_ca_row(self=model,
                                        axis=[int(input.Axis1()),int(input.Axis2())],
                                        color=input.RowTextActifColor(),
                                        color_sup = None,
                                        text_size = input.RowTextSize(),
                                        lim_contrib =input.RowLimContrib(),
                                        lim_cos2 = input.RowLimCos2(),
                                        title = input.RowTitle(),
                                        repel=input.RowPlotRepel(),
                                        ggtheme=pn.theme_gray())
                    else:
                        fig = fviz_ca_row(self=model,
                                        axis=[int(input.Axis1()),int(input.Axis2())],
                                        color=input.RowTextActifColor(),
                                        color_sup = input.RowTextSupColor(),
                                        text_size = input.RowTextSize(),
                                        lim_contrib =input.RowLimContrib(),
                                        lim_cos2 = input.RowLimCos2(),
                                        title = input.RowTitle(),
                                        repel=input.RowPlotRepel(),
                                        ggtheme=pn.theme_gray())
                    return fig
                elif input.RowTextColor() in ["cos2","contrib"]:
                    if model.row_sup is None:
                        fig = fviz_ca_row(self=model,
                                        axis=[int(input.Axis1()),int(input.Axis2())],
                                        color=input.RowTextColor(),
                                        color_sup = None,
                                        text_size = input.RowTextSize(),
                                        lim_contrib =input.RowLimContrib(),
                                        lim_cos2 = input.RowLimCos2(),
                                        title = input.RowTitle(),
                                        repel=input.RowPlotRepel(),
                                        ggtheme=pn.theme_gray())
                    else:
                        fig = fviz_ca_row(self=model,
                                        axis=[int(input.Axis1()),int(input.Axis2())],
                                        color=input.RowTextColor(),
                                        color_sup = input.RowTextSupColor(),
                                        text_size = input.RowTextSize(),
                                        lim_contrib =input.RowLimContrib(),
                                        lim_cos2 = input.RowLimCos2(),
                                        title = input.RowTitle(),
                                        repel=input.RowPlotRepel(),
                                        ggtheme=pn.theme_gray())
                    return fig

            # ------------------------------------------------------------------------------
            # Individual Factor Map - PCA
            @output
            @render.plot(alt="Rows Factor Map - CA")
            def RowFactorMap():
                return RowFactorPlot().draw()
            
            # Downlaod
            # @session.download(filename="Rows-Factor-Map.png")
            # def RowGraphDownloadPng():
            #     return pw.load_ggplot(RowFactorPlot()).savefig("Rows-Factor-Map.png")
            
            #--------------------------------------------------------------------------------
            # Reactive Columns Plot
            @reactive.Calc
            def ColFactorPlot():
                if input.ColTextColor() == "actif/sup":
                    if model.col_sup is None:
                        fig = fviz_ca_col(self=model,
                                        axis=[int(input.Axis1()),int(input.Axis2())],
                                        title=input.ColTitle(),
                                        color=input.ColTextActifColor(),
                                        color_sup=None,
                                        text_size=input.ColTextSize(),
                                        lim_contrib = input.ColLimContrib(),
                                        lim_cos2 = input.ColLimCos2(),
                                        repel=input.ColPlotRepel(),
                                        ggtheme=pn.theme_gray())
                    else:
                        fig = fviz_ca_col(self=model,
                                        axis=[int(input.Axis1()),int(input.Axis2())],
                                        title=input.ColTitle(),
                                        color=input.ColTextActifColor(),
                                        color_sup=input.ColTextSupColor(),
                                        text_size=input.ColTextSize(),
                                        lim_contrib = input.ColLimContrib(),
                                        lim_cos2 = input.ColLimCos2(),
                                        repel=input.ColPlotRepel(),
                                        ggtheme=pn.theme_gray())
                    
                    return fig
                elif input.ColTextColor() in ["cos2","contrib"]:
                    if model.col_sup is None:
                        fig = fviz_ca_col(self=model,
                                        axis=[int(input.Axis1()),int(input.Axis2())],
                                        title=input.ColTitle(),
                                        color=input.ColTextColor(),
                                        color_sup=None,
                                        text_size=input.ColTextSize(),
                                        lim_contrib = input.ColLimContrib(),
                                        lim_cos2 = input.ColLimCos2(),
                                        repel=input.ColPlotRepel(),
                                        ggtheme=pn.theme_gray())
                    else:
                        fig = fviz_ca_col(self=model,
                                        axis=[int(input.Axis1()),int(input.Axis2())],
                                        title=input.ColTitle(),
                                        color=input.ColTextColor(),
                                        color_sup=input.ColTextSupColor(),
                                        text_size=input.ColTextSize(),
                                        lim_contrib = input.ColLimContrib(),
                                        lim_cos2 = input.ColLimCos2(),
                                        repel=input.ColPlotRepel(),
                                        ggtheme=pn.theme_gray())
                    return fig

            # Variables Factor Map - PCA
            @output
            @render.plot(alt="Variables Factor Map - PCA")
            def ColFactorMap():
                return ColFactorPlot().draw()
            
            # Download button should be add
                
            #-------------------------------------------------------------------------------------------
            # Eigenvalue - Scree plot
            @output
            @render.plot(alt="Scree Plot - PCA")
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
            
            #############################################################################################
            #       Columns informations
            #############################################################################################
            #---------------------------------------------------------------------------------------------
            # Columns Coordinates
            @render.data_frame
            def ColCoordTable():
                ColCoord = model.col_["coord"].round(4).reset_index()
                ColCoord.columns = ["Columns", *ColCoord.columns[1:]]
                return DataTable(data=match_datalength(data=ColCoord,value=input.ColCoordLen()),filters=input.ColCoordFilter())
            
            #--------------------------------------------------------------------------------------------------
            # Columns Contributions
            @render.data_frame
            def ColContribTable():
                ColContrib = model.col_["contrib"].round(4).reset_index()
                ColContrib.columns = ["Columns", *ColContrib.columns[1:]]
                return  DataTable(data=match_datalength(data=ColContrib,value=input.ColContribLen()),filters=input.ColContribFilter())
            
            # Add Columns Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.ColContribGraphBtn)
            def _():
                GraphModalShow(text="Col",name="Contrib",max_axis=nbDim)

            # Reactive Columns Contributions Map
            @reactive.Calc
            def ColContribMap():
                fig = fviz_contrib(self=model,
                                   choice="col",
                                   axis=input.ColContribAxis(),
                                   top_contrib=int(input.ColContribTop()),
                                   color=input.ColContribColor(),
                                   bar_width=input.ColContribBarWidth(),
                                   ggtheme=pn.theme_gray())
                return fig
            
            # Plot columns Contributions
            @output
            @render.plot(alt="Columns Contributions Map - CA")
            def ColContribPlot():
                return ColContribMap().draw()
            
            #----- Download Button to add

            #--------------------------------------------------------------------------------------------
            # Columns Cos2
            @render.data_frame
            def ColCos2Table():
                ColCos2 = model.col_["cos2"].round(4).reset_index()
                ColCos2.columns = ["Columns", *ColCos2.columns[1:]]
                return  DataTable(data=match_datalength(data=ColCos2,value=input.ColCos2Len()),filters=input.ColCos2Filter())
            
            # Add Columns Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.ColCos2GraphBtn)
            def _():
                GraphModalShow(text="Col",name="Cos2",max_axis=nbDim)

            # Reactive Graph
            @reactive.Calc
            def ColCos2Map():
                fig = fviz_cos2(self=model,
                                choice="col",
                                axis=input.ColCos2Axis(),
                                top_cos2=int(input.ColCos2Top()),
                                color=input.ColCos2Color(),
                                bar_width=input.ColCos2BarWidth(),
                                ggtheme=pn.theme_gray())
                return fig
            
            # Plot Columns Cos2
            @output
            @render.plot(alt="Columns Cosines Map - CA")
            def ColCos2Plot():
                return ColCos2Map().draw()
            
            #---------------------------------------------------------------------------------
            ## Supplementary Columns
            #------------------------------------------------------------------------------------
            # Add supplementary Columns Conditional Panel
            @output
            @render.ui
            def ColSupPanel():
                return ui.panel_conditional("input.choice == 'ColSupRes'",
                            ui.br(),
                            ui.h5("Coordonnées"),
                            PanelConditional1(text="ColSup",name="Coord"),
                            ui.hr(),
                            ui.h5("Cos2 - Qualité de la représentation"),
                            PanelConditional1(text="ColSup",name="Cos2") 
                        )
            # Supplementary columns coordinates
            @render.data_frame
            def ColSupCoordTable():
                ColSupCoord = model.col_sup_["coord"].round(4).reset_index()
                ColSupCoord.columns = ["Columns", *ColSupCoord.columns[1:]]
                return DataTable(data=match_datalength(data=ColSupCoord,value=input.ColSupCoordLen()),filters=input.ColSupCoordFilter())
            
            # Supplementary columns Cos2
            @render.data_frame
            def ColSupCos2Table():
                ColSupCos2 = model.col_sup_["cos2"].round(4).reset_index()
                ColSupCos2.columns = ["Columns", *ColSupCos2.columns[1:]]
                return DataTable(data=match_datalength(data=ColSupCos2,value=input.ColSupCos2Len()),filters=input.ColSupCos2Filter())

            ############################################################################################
            #       Row Points Informations
            #############################################################################################

            #---------------------------------------------------------------------------------------------
            # Rows Coordinates
            @render.data_frame
            def RowCoordTable():
                RowCoord = model.row_["coord"].round(4).reset_index()
                RowCoord.columns = ["Rows", *RowCoord.columns[1:]]
                return DataTable(data = match_datalength(RowCoord,input.RowCoordLen()),filters=input.RowCoordFilter())
            
            #-------------------------------------------------------------------------------------------------
            # Rows Contributions
            @render.data_frame
            def RowContribTable():
                RowContrib = model.row_["contrib"].round(4).reset_index()
                RowContrib.columns = ["Rows", *RowContrib.columns[1:]]
                return  DataTable(data=match_datalength(RowContrib,input.RowContribLen()),filters=input.RowContribFilter())
            
            # Add rows Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.RowContribGraphBtn)
            def _():
                GraphModalShow(text="Row",name="Contrib",max_axis=nbDim)

            # Plot Rows Contributions
            @reactive.Calc
            def RowContribMap():
                fig = fviz_contrib(self=model,
                                   choice="row",
                                   axis=input.RowContribAxis(),
                                   top_contrib=int(input.RowContribTop()),
                                   color = input.RowContribColor(),
                                   bar_width= input.RowContribBarWidth(),
                                   ggtheme=pn.theme_gray())
                return fig

            @output
            @render.plot(alt="Rows Contributions Map - PCA")
            def RowContribPlot():
                return RowContribMap().draw()
            
            #----------------------------------------------------------------------------------------------------
            # Rows Cos2
            @render.data_frame
            def RowCos2Table():
                RowCos2 = model.row_["cos2"].round(4).reset_index()
                RowCos2.columns = ["Rows", *RowCos2.columns[1:]]
                return  DataTable(data = match_datalength(RowCos2,input.RowCos2Len()),filters=input.RowCos2Filter())
            
            # Add Rows Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.RowCos2GraphBtn)
            def _():
                GraphModalShow(text="Row",name="Cos2",max_axis=nbDim)

            # Plot Rows Cos2
            @reactive.Calc
            def RowCos2Map():
                fig = fviz_cos2(self=model,
                                choice="row",
                                axis=input.RowCos2Axis(),
                                top_cos2=int(input.RowCos2Top()),
                                color=input.RowCos2Color(),
                                bar_width=input.RowCos2BarWidth(),
                                ggtheme=pn.theme_gray())
                return fig
                
            @output
            @render.plot(alt="Rows Cosines Map - CA")
            def RowCos2Plot():
                return RowCos2Map().draw()
            
            #----------------------------------------------------------------------------
            # Supplementary Rows Coordinates
            @render.data_frame
            def RowSupCoordTable():
                RowSupCoord = model.row_sup_["coord"].round(4).reset_index()
                RowSupCoord.columns = ["Rows", *RowSupCoord.columns[1:]]
                return  DataTable(data = match_datalength(RowSupCoord,input.RowSupCoordLen()),filters=input.RowSupCoordFilter())
            
            # Supplementary Rows Cos2
            @render.data_frame
            def RowSupCos2Table():
                RowSupCos2 = model.row_sup_["cos2"].round(4).reset_index()
                RowSupCos2.columns = ["Rows", *RowSupCos2.columns[1:]]
                return  DataTable(data = match_datalength(RowSupCos2,input.RowSupCos2Len()),filters=input.RowSupCos2Filter())
            
            ##############################################################################################################
            ## Supplementary quantitatives variables informations
            ##------------------------------------------------------------------------------------------------------------
            # Add supplementary continuous/quantitatives Conditional Panel
            @output
            @render.ui
            def QuantiSupPanel():
                return ui.panel_conditional("input.choice == 'QuantiSupRes'",
                            ui.br(),
                            ui.h5("Coordonnées"),
                            PanelConditional1(text="QuantiSup",name="Coord"),
                            ui.hr(),
                            ui.h5("Cos2 - Qualité de la représentation"),
                            PanelConditional1(text="QuantiSup",name="Cos2")                   
                        )
            
            # Supplementary quantitatives variables coordinates
            @render.data_frame
            def QuantiSupCoordTable():
                QuantiSupCoord = model.quanti_sup_["coord"].round(4).reset_index()
                QuantiSupCoord.columns = ["Variables", *QuantiSupCoord.columns[1:]]
                return  DataTable(data = match_datalength(QuantiSupCoord,input.QuantiSupCoordLen()),filters=input.QuantiSupCoordFilter())
            
            # Supplementary quantitatives variables cos2
            @render.data_frame
            def QuantiSupCos2Table():
                QuantiSupCos2 = model.quanti_sup_["cos2"].round(4).reset_index()
                QuantiSupCos2.columns = ["Variables", *QuantiSupCos2.columns[1:]]
                return  DataTable(data = match_datalength(QuantiSupCos2,input.QuantiSupCos2Len()),filters=input.QuantiSupCos2Filter())
            
            #########################################################################################
            ## Supplementary qualitatives variables informations
            ##---------------------------------------------------------------------------------------
            # Add supplementary qualitatives Conditional Panel
            @output
            @render.ui
            def QualiSupPanel():
                return ui.panel_conditional("input.choice == 'QualiSupRes'",
                            ui.br(),
                            ui.h5("Coordonnées"),
                            PanelConditional1(text="QualiSup",name="Coord"),
                            ui.hr(),
                            ui.h5("Cos2 - Qualité de la représentation"),
                            PanelConditional1(text="QualiSup",name="Cos2"),
                            ui.hr(),
                            ui.h5("V-test"),
                            PanelConditional1(text="QualiSup",name="Vtest"),
                            ui.hr(),
                            ui.h5("Eta2 - Rapport de corrélation"),
                            PanelConditional1(text="QualiSup",name="Eta2"),               
                        )

            # Supplementary qualitatives coordinates
            @render.data_frame
            def QualiSupCoordTable():
                QualiSupCoord = model.quali_sup_["coord"].round(4).reset_index()
                QualiSupCoord.columns = ["Categories", *QualiSupCoord.columns[1:]]
                return  DataTable(data = match_datalength(QualiSupCoord,input.QualiSupCoordLen()),filters=input.QualiSupCoordFilter())
            
            # Supplementary qualitatives cos2
            @render.data_frame
            def QualiSupCos2Table():
                QualiSupCos2 = model.quali_sup_["cos2"].round(4).reset_index()
                QualiSupCos2.columns = ["Categories", *QualiSupCos2.columns[1:]]
                return  DataTable(data = match_datalength(QualiSupCos2,input.QualiSupCos2Len()),filters=input.QualiSupCos2Filter())
            
            # Supplementary qualitatives v-test
            @render.data_frame
            def QualiSupVtestTable():
                QualiSupVtest = model.quali_sup_["vtest"].round(4).reset_index()
                QualiSupVtest.columns = ["Categories", *QualiSupVtest.columns[1:]]
                return  DataTable(data = match_datalength(QualiSupVtest,input.QualiSupVtestLen()),filters=input.QualiSupVtestFilter())
            
            # Supplementary qualitatives Eta2
            @render.data_frame
            def QualiSupEta2Table():
                QualiSupEta2 = model.quali_sup_["eta2"].round(4).reset_index()
                QualiSupEta2.columns = ["Variables", *QualiSupEta2.columns[1:]]
                return  DataTable(data = match_datalength(QualiSupEta2,input.QualiSupEta2Len()),filters=input.QualiSupEta2Filter())

            #########################################################################################
            #       Description des axes
            #########################################################################################
            #---------------------------------------------------------------------------------------
            # Description of axis
            @output
            @render.ui
            def DimDesc():
                return ui.TagList(
                    ui.h5("Points lignes"),
                    PanelConditional1(text="Row",name="Desc"),
                    ui.hr(),
                    ui.h5("Points Colonnes"),
                    PanelConditional1(text="Col",name="Desc"),
                )
            
            @render.data_frame
            def RowDescTable():
                DimDesc = dimdesc(self=model,axis=None)
                if isinstance(DimDesc[input.Dimdesc()],dict):
                    DimDescRow = DimDesc[input.Dimdesc()]["row"].reset_index().rename(columns={"index":"Rows"})
                elif isinstance(DimDesc[input.Dimdesc()],pd.DataFrame):
                    DimDescRow = DimDesc[input.Dimdesc()].reset_index().rename(columns={"index":"Rows"})
                else:
                    DimDescRow = pd.DataFrame()
                return  DataTable(data = match_datalength(DimDescRow,input.RowDescLen()),filters=input.RowDescFilter())
            
            @render.data_frame
            def ColDescTable():
                DimDesc = dimdesc(self=model,axis=None)
                if isinstance(DimDesc[input.Dimdesc()],dict):
                    DimDescCol = DimDesc[input.Dimdesc()]["col"].reset_index().rename(columns={"index":"Columns"})
                else:
                    DimDescCol = pd.DataFrame()
                return  DataTable(data = match_datalength(DimDescCol,input.ColDescLen()),filters=input.ColDescFilter())
            
            ################################################################################################
            #       Résumé du jeu de données
            ###################################################################################################
            #-----------------------------------------------------------------------------------------------
            ### Distribution conditionelle (X/Y)
            @render.data_frame
            def CondDistOneTable():
                data = model.call_["X"].apply(lambda x : 100*x/np.sum(x),axis=0)
                data.loc["Total",:] = data.sum(axis=0)
                data = data.round(4).reset_index()
                return DataTable(data = match_datalength(data,input.CondDistOneLen()),filters=input.CondDistOneFilter())
            
            ### Distribution conditionelle (Y/X)
            @render.data_frame
            def CondDistTwoTable():
                data = model.call_["X"].apply(lambda x : 100*x/np.sum(x),axis=1)
                data.loc[:,"Total"] = data.sum(axis=1)
                data = data.round(4).reset_index()
                return DataTable(data = match_datalength(data,input.CondDistTwoLen()),filters=input.CondDistTwoFilter())
            
            #####################################################################################################
            #---------------------------------------------------------------------------------------------------
            # Overall Data
            @render.data_frame
            def OverallDataTable():
                overalldata = model.call_["Xtot"].reset_index()
                return DataTable(data = match_datalength(overalldata,input.OverallDataLen()),filters=input.OverallDataFilter())
            
        self.app_ui = app_ui
        self.app_server = server
    
    def run(self,**kwargs):

        """
        Run the app

        Parameters:
        ----------
        kwargs : objet = {}. See https://shiny.posit.co/py/api/App.html
        
        """

        app = App(ui=self.app_ui, server=self.app_server,static_assets=Path(__file__).parent / "www")
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