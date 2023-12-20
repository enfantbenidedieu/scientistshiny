# -*- coding: utf-8 -*-
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
import shinyswatch
from pathlib import Path
import pandas as pd
import patchworklib as pw
import plotnine as pn
import matplotlib.colors as mcolors
import nest_asyncio
import uvicorn

#  All scientisttools fonctions
from scientisttools.ggplot import (
    fviz_ca_row,
    fviz_ca_col,
    fviz_eig, 
    fviz_contrib,
    fviz_cosines,
    fviz_corrplot)
from scientisttools.extractfactor import (
    get_eig,
    get_ca_row,
    get_ca_col,
    dimdesc)
from scientistshiny.function import *

colors = mcolors.CSS4_COLORS
colors["cos2"] = "cos2"
colors["contrib"] = "contrib"

css_path = Path(__file__).parent / "www" / "style.css"

class CAshiny:
    """
    Correspondance Analysis (CA) with scientistshiny

    Description
    -----------
    Performs Correspondance Analysis (PCA) including supplementary row and/or column points on a Shiny application.
    Graphics can be downloaded in png, jpg and pdf.

    Usage
    -----
    CAshiny(fa_model)

    Parameters:
    ----------
    fa_model : An instance of class CA. A CA result from scientisttools.

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

    Author:
    -------
    Duvérier DJIFACK ZEBAZE : duverierdjifack@gmail.com

    Examples:
    ---------
    > from scientisttools.decomposition import PCA
    > from scientistshiny import CAshiny


    for jupyter notebooks
    https://stackoverflow.com/questions/74070505/how-to-run-fastapi-application-inside-jupyter
    """


    def __init__(self,fa_model=None):
        if fa_model.model_ != "ca":
            raise ValueError("Error : 'fa_model' must be an instance of class CA")
        
        # -----------------------------------------------------------------------------------
        # Initialise value choice
        value_choice = {"EigenRes": "Valeurs propres",
                        "ColRes"  : "Résultats pour les colonnes",
                        "RowRes"  : "Résultats pour les lignes"}
        if fa_model.row_sup_labels_ is not None:
            value_choice.update({"RowSupRes" : "Résultats pour les lignes supplémentaires"})
        if fa_model.col_sup_labels_ is not None:
            value_choice.update({"ColSupRes" : "Résultats pour les colonnes supplémentaires"})

        DimDescChoice = {}
        for i in range(min(3,fa_model.n_components_)):
            DimDescChoice.update({"Dim."+str(i+1) : "Dimension "+str(i+1)})

        # App UI
        app_ui = ui.page_fluid(
            ui.include_css(css_path),
            shinyswatch.theme.superhero(),
            ui.page_navbar(title=ui.div(ui.panel_title(ui.h2("Analyse des Correspondances"),window_title="CAshiny"),align="center"),inverse=True,id="navbar_id",padding={"style": "text-align: center;"}),
            ui.page_sidebar(
                ui.sidebar(
                    ui.panel_well(
                        ui.h6("Options graphiques",style="text-align : center"),
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
                    ui.nav("Graphes",
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
                        OverallPanelConditional(text="Col"),
                        OverallPanelConditional(text="Row"),
                        ui.output_ui("RowSupPanel"),
                        ui.output_ui("ColSupPanel")
                    ),
                    ui.nav("Description automatique des axes",
                        ui.input_radio_buttons(id="Dimdesc",
                                               label="Choisir les dimensions",
                                               choices=DimDescChoice,selected="Dim.1",inline=True),
                        ui.output_ui(id="DimDesc")
                    ),
                    ui.nav("Résumé du jeu de données",
                        PanelConditional1(text="ResumeData",name="")
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
            def RowTextChoice():
                if fa_model.row_sup_labels_ is not None:
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
                if fa_model.col_sup_labels_ is not None:
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
                            PanelConditional1(text="RowSup",name="Coord")
                        )
            
            # Add supplementary Columns Conditional Panel
            @output
            @render.ui
            def ColSupPanel():
                return ui.panel_conditional("input.choice == 'ColSupRes'",
                            ui.br(),
                            ui.h5("Coordonnées"),
                            PanelConditional1(text="ColSup",name="Coord")
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
                    if fa_model.row_sup_labels_ is None:
                        fig = fviz_ca_row(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            color=input.RowTextActifColor(),
                            color_sup = None,
                            text_size = input.RowTextSize(),
                            lim_contrib =input.RowLimContrib(),
                            lim_cos2 = input.RowLimCos2(),
                            title = input.RowTitle(),
                            repel=input.RowPlotRepel()
                        )
                    else:
                        fig = fviz_ca_row(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            color=input.RowTextActifColor(),
                            color_sup = input.RowTextSupColor(),
                            text_size = input.RowTextSize(),
                            lim_contrib =input.RowLimContrib(),
                            lim_cos2 = input.RowLimCos2(),
                            title = input.RowTitle(),
                            repel=input.RowPlotRepel()
                        )
                    return fig
                elif input.RowTextColor() in ["cos2","contrib"]:
                    if fa_model.row_sup_labels_ is None:
                        fig = fviz_ca_row(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            color=input.RowTextColor(),
                            color_sup = None,
                            text_size = input.RowTextSize(),
                            lim_contrib =input.RowLimContrib(),
                            lim_cos2 = input.RowLimCos2(),
                            title = input.RowTitle(),
                            repel=True
                        )
                    else:
                        fig = fviz_ca_row(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            color=input.RowTextColor(),
                            color_sup = input.RowTextSupColor(),
                            text_size = input.RowTextSize(),
                            lim_contrib =input.RowLimContrib(),
                            lim_cos2 = input.RowLimCos2(),
                            title = input.RowTitle(),
                            repel=input.RowPlotRepel()
                        )
                    return fig

            # ------------------------------------------------------------------------------
            # Individual Factor Map - PCA
            @output
            @render.plot(alt="Rows Factor Map - CA")
            def RowFactorMap():
                return RowFactorPlot().draw()
            
            # Downlaod
            @session.download(filename="Rows-Factor-Map.png")
            def RowGraphDownloadPng():
                return pw.load_ggplot(RowFactorPlot()).savefig("Rows-Factor-Map.png")
            
            #--------------------------------------------------------------------------------
            # Reactive Columns Plot
            @reactive.Calc
            def ColFactorPlot():
                if input.ColTextColor() == "actif/sup":
                    if fa_model.col_sup_labels_ is None:
                        fig = fviz_ca_col(
                        self=fa_model,
                        axis=[int(input.Axis1()),int(input.Axis2())],
                        title=input.ColTitle(),
                        color=input.ColTextActifColor(),
                        color_sup=None,
                        text_size=input.ColTextSize(),
                        lim_contrib = input.ColLimContrib(),
                        lim_cos2 = input.ColLimCos2(),
                        repel=input.ColPlotRepel()
                        )
                    else:
                        fig = fviz_ca_col(
                        self=fa_model,
                        axis=[int(input.Axis1()),int(input.Axis2())],
                        title=input.ColTitle(),
                        color=input.ColTextActifColor(),
                        color_sup=input.ColTextSupColor(),
                        text_size=input.ColTextSize(),
                        lim_contrib = input.ColLimContrib(),
                        lim_cos2 = input.ColLimCos2(),
                        repel=input.ColPlotRepel()
                        )
                    
                    return fig
                elif input.ColTextColor() in ["cos2","contrib"]:
                    if fa_model.col_sup_labels_ is None:
                        fig = fviz_ca_col(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            title=input.ColTitle(),
                            color=input.ColTextColor(),
                            color_sup=None,
                            text_size=input.ColTextSize(),
                            lim_contrib = input.ColLimContrib(),
                            lim_cos2 = input.ColLimCos2(),
                            repel=input.ColPlotRepel()
                            )
                    else:
                        fig = fviz_ca_col(
                            self=fa_model,
                            axis=[int(input.Axis1()),int(input.Axis2())],
                            title=input.ColTitle(),
                            color=input.ColTextColor(),
                            color_sup=input.ColTextSupColor(),
                            text_size=input.ColTextSize(),
                            lim_contrib = input.ColLimContrib(),
                            lim_cos2 = input.ColLimCos2(),
                            repel=input.ColPlotRepel()
                        )
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
            
            #############################################################################################
            #       Columns informations
            #############################################################################################
            #---------------------------------------------------------------------------------------------
            # Columns Coordinates
            @output
            @render.data_frame
            def ColCoordTable():
                ColCoord = get_ca_col(fa_model)["coord"].round(4).reset_index().rename(columns={"index" : "Columns"})
                return DataTable(data=match_datalength(data=ColCoord,value=input.ColCoordLen()),
                                filters=input.ColCoordFilter())
            
            #--------------------------------------------------------------------------------------------------
            # Columns Contributions
            @output
            @render.data_frame
            def ColContribTable():
                ColContrib = get_ca_col(fa_model)["contrib"].round(4).reset_index().rename(columns={"index" : "Columns"})
                return  DataTable(data=match_datalength(data=ColContrib,value=input.ColContribLen()),
                                  filters=input.ColContribFilter())
            
            # Add Columns Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.ColContribGraphBtn)
            def _():
                GraphModalShow(text="Col",name="Contrib")

            # Reactive Columns Contributions Map
            @reactive.Calc
            def ColContribMap():
                fig = fviz_contrib(
                    self=fa_model,
                    choice="var",
                    axis=input.ColContribAxis(),
                    top_contrib=int(input.ColContribTop()),
                    color=input.ColContribColor(),
                    bar_width=input.ColContribBarWidth()
                    )
                return fig
            
            # Plot columns Contributions
            @output
            @render.plot(alt="Columns Contributions Map - CA")
            def ColContribPlot():
                return ColContribMap().draw()
            
            #----- Download Button

            # Add Variables Contributions/Correlations Modal Show
            @reactive.Effect
            @reactive.event(input.ColContribCorrGraphBtn)
            def _():
                GraphModelModal2(text="Col",name="Contrib",title=None)

            # Reactive columns contribution/correlations Map
            @reactive.Calc
            def ColContribCorrMap():
                fig = fviz_corrplot(
                    X=get_ca_col(fa_model)["contrib"],
                    title=input.ColContribCorrTitle(),
                    outline_color=input.ColContribCorrColor(),
                    xlabel="Columns",
                    colors=[input.ColContribCorrLowColor(),
                            input.ColContribCorrMidColor(),
                            input.ColContribCorrHightColor()
                            ])+pn.theme_gray()
                return fig

            # Plot Columns Contributions
            @output
            @render.plot(alt="Columns Contributions/Correlations Map - CA")
            def ColContribCorrPlot():
                return ColContribCorrMap().draw()
            
            # Download button

            #--------------------------------------------------------------------------------------------
            # Columns Cos2
            @output
            @render.data_frame
            def ColCos2Table():
                ColCos2 = get_ca_col(fa_model)["cos2"].round(4).reset_index().rename(columns={"index" : "Columns"})
                return  DataTable(data=match_datalength(data=ColCos2,value=input.ColCos2Len()),
                                  filters=input.ColCos2Filter())
            
            # Add Columns Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.ColCos2GraphBtn)
            def _():
                GraphModalShow(text="Col",name="Cos2")

            # Reactive Graph
            @reactive.Calc
            def ColCos2Map():
                fig = fviz_cosines(
                    self=fa_model,
                    choice="var",
                    axis=input.ColCos2Axis(),
                    top_cos2=int(input.ColCos2Top()),
                    color=input.ColCos2Color(),
                    bar_width=input.ColCos2BarWidth()
                )
                return fig
            
            # Plot Columns Cos2
            @output
            @render.plot(alt="Columns Cosines Map - CA")
            def ColCos2Plot():
                return ColCos2Map().draw()
            
            #-----------------------------------------------------------------------------
            # Add Columns Cosinus Correlation Modal Show
            @reactive.Effect
            @reactive.event(input.ColCos2CorrGraphBtn)
            def _():
                GraphModelModal2(text="Col",name="Cos2",title=None)
        
            @reactive.Calc
            def ColCos2CorrMap():
                fig = fviz_corrplot(
                        X=get_ca_col(fa_model)["cos2"],
                        title=input.ColCos2CorrTitle(),
                        outline_color=input.ColCos2CorrColor(),
                        xlabel="Columns",
                        colors=[input.ColCos2CorrLowColor(),
                                input.ColCos2CorrMidColor(),
                                input.ColCos2CorrHightColor()
                                ])+pn.theme_gray()
                return fig

            # Plot Columns Cos2/Correlations plot
            @output
            @render.plot(alt="Columns Cosinus/Correlations Map - CA")
            def ColCos2CorrPlot():
                return ColCos2CorrMap().draw()

            #---------------------------------------------------------------------------------
            ## Supplementary Columns
            @output
            @render.data_frame
            def ColSupCoordTable():
                colsupcoord = pd.DataFrame(get_ca_col(fa_model)["col_sup"]["coord"],
                                           index=fa_model.col_sup_labels_,
                                           columns=fa_model.dim_index_)
                ColSupCoord = colsupcoord.round(4).reset_index().rename(columns={"index" : "Columns"})
                return DataTable(data=match_datalength(data=ColSupCoord,value=input.ColSupCoordLen()),
                                 filters=input.ColSupCoordFilter())

            ############################################################################################
            #       Row Points Informations
            #############################################################################################

            #---------------------------------------------------------------------------------------------
            # Rows Coordinates
            @output
            @render.data_frame
            def RowCoordTable():
                RowCoord = get_ca_row(fa_model)["coord"].round(4).reset_index()
                return DataTable(data = match_datalength(RowCoord,input.RowCoordLen()),
                                 filters=input.RowCoordFilter())
            
            #-------------------------------------------------------------------------------------------------
            # Rows Contributions
            @output
            @render.data_frame
            def RowContribTable():
                RowContrib = get_ca_row(fa_model)["contrib"].round(4).reset_index()
                return  DataTable(data=match_datalength(RowContrib,input.RowContribLen()),
                                  filters=input.RowContribFilter())
            
            # Add rows Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.RowContribGraphBtn)
            def _():
                GraphModalShow(text="Row",name="Contrib")

            # Plot Rows Contributions
            @reactive.Calc
            def RowContribMap():
                fig = fviz_contrib(
                    self=fa_model,
                    choice="ind",
                    axis=input.RowContribAxis(),
                    top_contrib=int(input.RowContribTop()),
                    color = input.RowContribColor(),
                    bar_width= input.RowContribBarWidth())
                return fig

            @output
            @render.plot(alt="Rows Contributions Map - PCA")
            def RowContribPlot():
                return RowContribMap().draw()
            
            # Add Rows Contributions/Correlation Modal Show
            @reactive.Effect
            @reactive.event(input.RowContribCorrGraphBtn)
            def _():
                GraphModelModal2(text="Row",name="Contrib",title=None)

            # Plot Row Contributions
            @reactive.Calc
            def RowContribCorrMap():
                fig = fviz_corrplot(
                    X=get_ca_row(fa_model)["contrib"],
                    title=input.RowContribCorrTitle(),
                    xlabel="Rows",
                    outline_color=input.RowContribCorrColor(),
                    colors=[input.RowContribCorrLowColor(),
                            input.RowContribCorrMidColor(),
                            input.RowContribCorrHightColor()
                            ])+pn.theme_gray()
                return fig

            @output
            @render.plot(alt="Rows Contributions/Correlations Map - PCA")
            def RowContribCorrPlot():
                return RowContribCorrMap().draw()
            
            #----------------------------------------------------------------------------------------------------
            # Rows Cos2 
            @output
            @render.data_frame
            def RowCos2Table():
                RowCos2 = get_ca_row(fa_model)["cos2"].round(4).reset_index()
                return  DataTable(data = match_datalength(RowCos2,input.RowCos2Len()),
                                filters=input.RowCos2Filter())
            
            # Add Rows Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.RowCos2GraphBtn)
            def _():
                GraphModalShow(text="Row",name="Cos2")

            # Plot Rows Cos2
            @reactive.Calc
            def RowCos2Map():
                fig = fviz_cosines(
                    self=fa_model,
                    choice="ind",
                    axis=input.RowCos2Axis(),
                    top_cos2=int(input.RowCos2Top()),
                    color=input.RowCos2Color(),
                    bar_width=input.RowCos2BarWidth())
                return fig
                
            @output
            @render.plot(alt="Rows Cosines Map - CA")
            def RowCos2Plot():
                return RowCos2Map().draw()
                
            # Add Columns Cosines/Correlation Modal Show
            @reactive.Effect
            @reactive.event(input.RowCos2CorrGraphBtn)
            def _():
                GraphModelModal2(text="Row",name="Cos2",title=None)

            # Plot Rows Contributions
            @reactive.Calc
            def RowCos2CorrMap():
                fig = fviz_corrplot(
                    X=get_ca_col(fa_model)["cos2"],
                    title=input.RowCos2CorrTitle(),
                    xlabel="Rows",
                    outline_color=input.RowCos2CorrColor(),
                    colors=[input.RowCos2CorrLowColor(),
                            input.RowCos2CorrMidColor(),
                            input.RowCos2CorrHightColor()
                        ])+pn.theme_gray()
                return fig

            @output
            @render.plot(alt="Rows Cosinus/Correlations Map - CA")
            def RowCos2CorrPlot():
                return RowCos2CorrMap().draw()
            
            #----------------------------------------------------------------------------
            # Supplementary Rows Coordinates
            @output
            @render.data_frame
            def RowSupCoordTable():
                RowSupCoord = get_ca_row(fa_model)["row_sup"]["coord"].round(4).reset_index()
                return  DataTable(data = match_datalength(RowSupCoord,input.RowSupCoordLen()),
                                filters=input.RowSupCoordFilter())
            
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
            
            @output
            @render.data_frame
            def RowDescTable():
                DimDesc = dimdesc(self=fa_model,axis=None)
                if isinstance(DimDesc[input.Dimdesc()],dict):
                    DimDescRow = DimDesc[input.Dimdesc()]["row"].reset_index().rename(columns={"index":"Rows"})
                elif isinstance(DimDesc[input.Dimdesc()],pd.DataFrame):
                    DimDescRow = DimDesc[input.Dimdesc()].reset_index().rename(columns={"index":"Rows"})
                else:
                    DimDescRow = pd.DataFrame()
                return  DataTable(data = match_datalength(DimDescRow,input.RowDescLen()),
                                  filters=input.RowDescFilter())
            
            @output
            @render.data_frame
            def ColDescTable():
                DimDesc = dimdesc(self=fa_model,axis=None)
                if isinstance(DimDesc[input.Dimdesc()],dict):
                    DimDescCol = DimDesc[input.Dimdesc()]["col"].reset_index().rename(columns={"index":"Columns"})
                else:
                    DimDescCol = pd.DataFrame()
                return  DataTable(data = match_datalength(DimDescCol,input.ColDescLen()),
                                  filters=input.ColDescFilter())
            
            ################################################################################################
            #       Résumé du jeu de données
            ###################################################################################################
            #-----------------------------------------------------------------------------------------------
            ### Statistiques descriptives
            @output
            @render.data_frame
            def ResumeDataTable():
                StatsDesc = fa_model.active_data_.describe(include="all").round(4).T.reset_index().rename(columns={"index":"Variables"})
                return  DataTable(data = match_datalength(StatsDesc,input.ResumeDataLen()),
                                  filters=input.ResumeDataFilter())
            
            #####################################################################################################
            #---------------------------------------------------------------------------------------------------
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
    
    def stop(self):
        """
        
        
        """
        app = App(ui=self.app_ui, server=self.server)
        return app.stop()


