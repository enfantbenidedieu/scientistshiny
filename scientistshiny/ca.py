# -*- coding: utf-8 -*-
from shiny import Inputs, Outputs, Session, render, ui, reactive
import shinyswatch
import numpy as np
import pandas as pd
import plotnine as pn
import matplotlib.colors as mcolors
from scientisttools import CA,fviz_ca_row,fviz_ca_col,fviz_eig, fviz_contrib,fviz_cos2, fviz_corrcircle
from scientistshiny.base import Base
from scientistshiny.function import *

class CAshiny(Base):
    """
    Correspondance Analysis (CA) with scientistshiny
    ------------------------------------------------

    Description
    -----------
    Performs Correspondance Analysis (CA) with supplementary elements (rows points and/or columns points and/or quantitative variables and/or qualitative variables) on a Shiny for Python application. Allows to change CA graphical parameters. Graphics can be downloaded in png, jpg and pdf.

    Usage
    -----
    ```python
    >>> CAshiny(model)
    ```

    Parameters
    ----------
    `model`: a pandas dataframe or a table with n rows and p columns, i.e a contingency table or an object of class CA (a CA result from scientisttools).

    Returns
    -------
    `Graphs` : a tab containing the the row and column points factor map (with supplementary columns and supplementary rows)

    `Values` : a tab containing the eigenvalues, the results for the columns, for the rows, for the supplementary elements (rows points and/or columns points and/or quantitative variables and/or qualitative variables).

    `Summary of dataset` : a tab containing the summary of the dataset.

    `Data` : a tab containing the dataset with a nice display.

    The left part of the application allows to change some elements of the graphs (axes, variables, colors,.)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com

    Examples
    --------
    ```python
    >>> # Load dataset and functions
    >>> import numpy as np
    >>> from scientisttools import CA, load_housetasks, load_children
    >>> from scientistshiny import CAshiny
    >>> 
    >>> # CA with scientistshiny
    >>> housetasks = load_housetasks()
    >>> res_shiny = CAshiny(model=housetasks)
    >>> res_shiny.run()
    >>> 
    >>> # CAshiny on a result of a CA
    >>> children = load_children()
    >>> children["group"] = ["A"]*4 + ["B"]*5 + ["C"]*5 +[np.nan]*4
    >>> res_ca = CA(n_components=None,row_sup=list(range(14,18)),col_sup=list(range(5,8)),quali_sup=8).fit(children)
    >>> res_shiny = CAshiny(model=res_ca)
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
                quali_sup = [model.columns.tolist().index(x) for x in model.columns if x in is_quali.columns]
            else:
                quali_sup = None
            
            # Fit the CA with scientisttools
            model = CA(quali_sup=quali_sup).fit(model)

        # Check if model is Correspondence Analysis (CA)
        if model.model_ != "ca":
            raise TypeError("'model' must be an object of class CA")
        
        graph_choices = {"fviz_row": "Points lignes","fviz_col" : "Points colonnes"}

        row_text_color_choices = {"actif/sup": "actifs/supplémentaires","cos2":"Cosinus","contrib":"Contribution"}

        # Initialise value choice
        value_choices = {"eigen_res":"Valeurs propres","col_res":"Résultats pour les colonnes","row_res":"Résultats pour les lignes"}

        # Resume choices
        resumes_choices = {"x/y":"Distributions conditionelles (X/Y)","y/x":"Distributions conditionnelles (Y/X)"}

        # Update check if supplementary rows
        if hasattr(model,"row_sup_"):
            value_choices = {**value_choices,**{"row_sup_res":"Résultats pour les lignes supplémentaires"}}

        # Check if supplementary columns
        if hasattr(model,"col_sup_"):
            value_choices = {**value_choices,**{"col_sup_res":"Résultats pour les colonnes supplémentaires"}}

        # Check if supplementary quantitatives columns
        if hasattr(model,"quanti_sup_"):
            graph_choices = {**graph_choices,"fviz_quanti_sup" : "Variables quantitatives"}
            row_text_color_choices = {**row_text_color_choices,**{"var_quant" : "Variable quantitative"}}
            value_choices = {**value_choices,**{"quanti_sup_res":"Résultats pour les variables quantitatives supplémentaires"}}
            resumes_choices = {**resumes_choices,**{"hist" : "Histogramme"}}

        # Check if supplementary qualitatives columns
        if hasattr(model,"quali_sup_"):
            row_text_color_choices = {**row_text_color_choices,**{"var_qual" : "Variable qualitative"}}
            value_choices = {**value_choices,**{"quali_sup_res":"Résultats pour les variables qualitatives supplémentaires"}}
            resumes_choices = {**resumes_choices,**{"bar_plot" : "Diagramme en barres"}}

        # App UI
        app_ui = ui.page_fluid(
            ui.include_css(css_path),
            shinyswatch.theme.superhero(),
            header(title="Analyse Factorielle des Correspondances",model_name="CA"),
            ui.page_sidebar(
                ui.sidebar(
                    ui.panel_well(
                        ui.h6("Options graphiques",style="text-align : center"),
                        ui.div(ui.h6("Axes"),style="display: inline-block;padding: 5px"),
                        axes_input_select(model=model),
                        ui.br(),
                        ui.div(ui.input_select(id="fviz_choice",label="",choices=graph_choices,selected="fviz_row",multiple=False,width="100%")),
                        ui.panel_conditional("input.fviz_choice === 'fviz_row'",
                            title_input(id="row_title",value="Row points - CA"),
                            text_size_input(which="row"),
                            point_select_input(id="row_point_select"),
                            ui.panel_conditional("input.row_point_select === 'cos2'",ui.div(lim_cos2(id="row_lim_cos2"),align="center")),
                            ui.panel_conditional("input.row_point_select === 'contrib'",ui.div(lim_contrib(id="row_lim_contrib"),align="center")),
                            text_color_input(id="row_text_color",choices=row_text_color_choices),
                            ui.panel_conditional("input.row_text_color === 'actif/sup'",
                                ui.input_select(id="row_text_actif_color",label="Points lignes actifs",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                                ui.output_ui("row_text_sup"),
                                ui.output_ui("row_text_quali_sup")
                            ),
                            ui.panel_conditional("input.row_text_color === 'var_quant'",ui.output_ui("row_text_var_quant")),
                            ui.panel_conditional("input.row_text_color === 'var_qual'",ui.output_ui("row_text_var_qual")),
                            ui.input_switch(id="row_plot_repel",label="repel",value=True)
                        ),
                        ui.panel_conditional("input.fviz_choice ==='fviz_col'",
                            title_input(id="col_title",value="Columns points - CA"),
                            text_size_input(which="col"),
                            point_select_input(id="col_point_select"),
                            ui.panel_conditional("input.col_point_select === 'cos2'",ui.div(lim_cos2(id="col_lim_cos2"),align="center")),
                            ui.panel_conditional("input.col_point_select === 'contrib'",ui.div(lim_contrib(id="col_lim_contrib"),align="center")),
                            text_color_input(id="col_text_color",choices={"actif/sup" : "actif/supplémentaire","cos2":"Cosinus","contrib":"Contribution"}),
                            ui.panel_conditional("input.col_text_color ==='actif/sup'",
                                ui.input_select(id="col_text_actif_color",label="Points colonnes actives",choices={x:x for x in mcolors.CSS4_COLORS},selected="black",multiple=False,width="100%"),
                                ui.output_ui("col_text_sup")
                            ),
                            ui.input_switch(id="col_plot_repel",label="repel",value=True)
                        ),
                        ui.output_ui("quanti_sup_fviz")
                    ),
                    ui.div(ui.input_action_button(id="exit",label="Quitter l'application",style='padding:5px; background-color: #fcac44;text-align:center;white-space: normal;'),align="center"),
                    width="25%"
                ),
                ui.navset_card_tab(
                    ui.nav_panel("Graphes",
                        ui.row(
                            ui.column(6,
                                ui.div(ui.output_plot("fviz_row_plot",width='100%',height="600px",fill=True),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px"),
                                ui.div(ui.download_button(id="download_row_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_row_plot_png",label="png",style = download_btn_style),style="display: inline-block;"),
                                ui.div(ui.download_button(id="download_row_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                                align="center"
                            ),
                            ui.column(6,
                                ui.div(ui.output_plot("fviz_col_plot",width='100%',height="600px"),align="center"),
                                ui.hr(),
                                ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                                ui.div(ui.download_button(id="download_col_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;",align="center"),
                                ui.div(ui.download_button(id="download_col_plot_png",label="png",style = download_btn_style),style="display: inline-block;",align="center"),
                                ui.div(ui.download_button(id="download_col_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;",align="center"),
                                align="center"
                            )
                        ),
                        ui.output_plot("quanti_sup_plot")
                    ),
                    ui.nav_panel("Valeurs",
                        ui.input_radio_buttons(id="value_choice",label=ui.h6("Quelles sorties voulez-vous?"),choices=value_choices,inline=True),
                        ui.br(),
                        eigen_panel(),
                        ui.panel_conditional("input.value_choice === 'col_res'",
                            ui.input_radio_buttons(id="col_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","contrib":"Contributions","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                            ui.panel_conditional("input.col_choice === 'coord'",panel_conditional1(text="col",name="coord")),
                            ui.panel_conditional("input.col_choice === 'contrib'",panel_conditional2(text="col",name="contrib")),
                            ui.panel_conditional("input.col_choice === 'cos2'",panel_conditional2(text="col",name="cos2"))
                        ),
                        ui.panel_conditional("input.value_choice === 'row_res'",
                            ui.input_radio_buttons(id="row_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","contrib":"Contributions","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                            ui.panel_conditional("input.row_choice === 'coord'",panel_conditional1(text="row",name="coord")),
                            ui.panel_conditional("input.row_choice === 'contrib'",panel_conditional2(text="row",name="contrib")),
                            ui.panel_conditional("input.row_choice === 'cos2'",panel_conditional2(text="row",name="cos2"))
                        ),
                        ui.output_ui("row_sup_panel"),
                        ui.output_ui("col_sup_panel"),
                        ui.output_ui("quanti_sup_panel"),
                        ui.output_ui("quali_sup_panel")
                    ),
                    ui.nav_panel("Résumé du jeu de données",
                        ui.input_radio_buttons(id="resume_choice",label=ui.h6("Quel type de distributions?"),choices=resumes_choices,selected="x/y",width="100%",inline=True),
                        ui.br(),
                        ui.panel_conditional("input.resume_choice === 'x/y'",panel_conditional1(text="cond_dist",name="one")),
                        ui.panel_conditional("input.resume_choice === 'y/x'",panel_conditional1(text="cond_dist",name="two")),
                        ui.output_ui("quali_sup_graph")
                    ),
                    ui.nav_panel("Données",panel_conditional1(text="overall",name="data"))
                )
            )
        )

        def server(input:Inputs, output:Outputs, session:Session):

            #----------------------------------------------------------------------------------------------
            # Disable x and y axis
            @reactive.Effect
            def _():
                x = int(input.axis1())
                Dim = [i for i in range(model.call_["n_components"]) if i > x]
                ui.update_select(id="axis2",label="",choices={x : x for x in Dim},selected=Dim[0])

            @reactive.Effect
            def _():
                x = int(input.axis2())
                Dim = [i for i in range(model.call_["n_components"]) if i < x]
                ui.update_select(id="axis1",label="",choices={x : x for x in Dim},selected=Dim[0])

            #--------------------------------------------------------------------------------------------------
            # Add supplementary rows color choice
            if hasattr(model,"row_sup_"):
                @render.ui
                def row_text_sup():
                    return ui.TagList(ui.input_select(id="row_text_sup_color",label="Points lignes supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="blue",multiple=False,width="100%"))
            
            #---------------------------------------------------------------------------------------------------
            # Add supplementary qualitative columns color choice
            if hasattr(model,"quali_sup_"):
                @render.ui
                def row_text_quali_sup():
                    return ui.TagList(ui.input_select(id="row_text_quali_sup_color",label="Modalités supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="red",multiple=False,width="100%"))

            #--------------------------------------------------------------------------------------------------------------
            # Disable rows colors
            if hasattr(model,"row_sup_") and hasattr(model,"quali_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="row_text_actif_color",label="Points lignes actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.row_text_sup_color(),input.row_text_quali_sup_color()]]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="row_text_sup_color",label="Points lignes supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.row_text_actif_color(),input.row_text_quali_sup_color()]]},selected="blue")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="row_text_quali_sup_color",label="Modalités supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i not in [input.row_text_actif_color(),input.row_text_sup_color()]]},selected="red")
            elif hasattr(model,"row_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="row_text_actif_color",label="Points lignes actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.row_text_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="row_text_sup_color",label="Points lignes supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.row_text_actif_color()]},selected="blue")
            elif hasattr(model,"quali_sup_"):
                @reactive.Effect
                def _():
                    ui.update_select(id="row_text_actif_color",label="Points lignes actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.row_text_quali_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="row_text_quali_sup_color",label="Modalités supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.row_text_actif_color()]},selected="red")

            #---------------------------------------------------------------------------------------------------------------
            if hasattr(model,"quanti_sup_"):
                @render.ui
                def row_text_var_quant():
                    quanti_sup_labels = model.quanti_sup_["coord"].index.tolist()
                    return ui.TagList(ui.input_select(id="row_text_var_quant_color",label="Choix de la variable",choices={x:x for x in quanti_sup_labels},selected=quanti_sup_labels[0],multiple=False))
                
            #-------------------------------------------------------------------------------------------------
            if hasattr(model,"quali_sup_"):
                @render.ui
                def row_text_var_qual():
                    quali_sup_labels = model.quali_sup_["eta2"].index.tolist()
                    return ui.TagList(
                            ui.input_select(id="row_text_var_qual_color",label="Choix de la variable",choices={x:x for x in quali_sup_labels},selected=quali_sup_labels[0],multiple=False,width="100%"),
                            ui.input_switch(id="row_text_add_ellipse",label="Trace les ellipses de confiance autour des modalités",value=False)
                        )
            
            #-----------------------------------------------------------------------------------------------
            if hasattr(model,"col_sup_"):
                @render.ui
                def col_text_sup():
                    return ui.TagList(ui.input_select(id="col_text_sup_color",label="Points colonnes supplémentaires",choices={x:x for x in mcolors.CSS4_COLORS},selected="blue",multiple=False,width="100%"))

                # Disable actifs and supplementary columns colors
                @reactive.Effect
                def _():
                    ui.update_select(id="col_text_actif_color",label="Points colonnes actifs",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.col_text_sup_color()]},selected="black")
                
                @reactive.Effect
                def _():
                    ui.update_select(id="col_text_sup_color",label="Points colonnes supplémentaires",choices={x:x for x in [i for i in mcolors.CSS4_COLORS if i != input.col_text_actif_color()]},selected="blue")

            #--------------------------------------------------------------------------------
            ## Rows plot
            #---------------------------------------------------------------------------------
            # Reactive rows plot
            @reactive.Calc
            def plot_row():
                if hasattr(model,"row_sup_"):
                    row_sup = True
                else:
                    row_sup = False
                
                if hasattr(model,"quali_sup_"):
                    quali_sup = True
                else:
                    quali_sup = False
                
                if input.row_text_color() == "actif/sup":
                    if hasattr(model,"row_sup_"):
                        color_sup = input.row_text_sup_color()
                    else:
                        color_sup = None
                    
                    if hasattr(model,"quali_sup_"):
                        color_quali_sup = input.row_text_quali_sup_color()
                    else:
                        color_quali_sup = None
                    
                    fig = fviz_ca_row(self=model,
                                      axis=[int(input.axis1()),int(input.axis2())],
                                      color=input.row_text_actif_color(),
                                      row_sup=row_sup,
                                      color_sup = color_sup,
                                      quali_sup=quali_sup,
                                      color_quali_sup=color_quali_sup,
                                      text_size = input.row_text_size(),
                                      lim_contrib =input.row_lim_contrib(),
                                      lim_cos2 = input.row_lim_cos2(),
                                      title = input.row_title(),
                                      repel=input.row_plot_repel())
                elif input.row_text_color() in ["cos2","contrib"]:
                    fig = fviz_ca_row(self=model,
                                      axis=[int(input.axis1()),int(input.axis2())],
                                      color=input.row_text_color(),
                                      row_sup=row_sup,
                                      quali_sup=quali_sup,
                                      text_size = input.row_text_size(),
                                      lim_contrib =input.row_lim_contrib(),
                                      lim_cos2 = input.row_lim_cos2(),
                                      title = input.row_title(),
                                      repel=input.row_plot_repel())
                elif input.row_text_color() == "var_quant":
                    fig = fviz_ca_row(self=model,
                                      axis=[int(input.axis1()),int(input.axis2())],
                                      color=input.row_text_var_quant_color(),
                                      row_sup=row_sup,
                                      quali_sup=quali_sup,
                                      text_size = input.row_text_size(),
                                      lim_contrib =input.row_lim_contrib(),
                                      lim_cos2 = input.row_lim_cos2(),
                                      title = input.row_title(),
                                      legend_title=input.row_text_var_quant_color(),
                                      repel=input.row_plot_repel())
                elif input.row_text_color() == "var_qual":
                    fig = fviz_ca_row(self=model,
                                      axis=[int(input.axis1()),int(input.axis2())],
                                      habillage=input.row_text_var_qual_color(),
                                      add_ellipses=input.row_text_add_ellipse(),
                                      row_sup=row_sup,
                                      quali_sup=quali_sup,
                                      text_size = input.row_text_size(),
                                      lim_contrib =input.row_lim_contrib(),
                                      lim_cos2 = input.row_lim_cos2(),
                                      title = input.row_title(),
                                      repel=input.row_plot_repel())
                return fig+pn.theme_gray()

            # Render Rows plot
            @render.plot(alt="Rows Factor Map - CA")
            def fviz_row_plot():
                return plot_row().draw()

            # Downlaod
            # @session.download(filename="Rows-Factor-Map.png")
            # def RowGraphDownloadPng():
            #     return pw.load_ggplot(RowFactorPlot()).savefig("Rows-Factor-Map.png")

            #--------------------------------------------------------------------------------
            ## Columns plot
            #--------------------------------------------------------------------------------
            # Reactive Columns Plot
            @reactive.Calc
            def plot_col():
                if hasattr(model,"col_sup_"):
                    col_sup = True
                else:
                    col_sup = False

                if input.col_text_color() == "actif/sup":
                    if hasattr(model,"col_sup_"):
                        color_sup = input.col_text_sup_color()
                    else:
                        color_sup = None

                    fig = fviz_ca_col(self=model,
                                      axis=[int(input.axis1()),int(input.axis2())],
                                      title=input.col_title(),
                                      color=input.col_text_actif_color(),
                                      col_sup=col_sup,
                                      color_sup=color_sup,
                                      text_size=input.col_text_size(),
                                      lim_contrib = input.col_lim_contrib(),
                                      lim_cos2 = input.col_lim_cos2(),
                                      repel=input.col_plot_repel())
                elif input.col_text_color() in ["cos2","contrib"]:
                    fig = fviz_ca_col(self=model,
                                      axis=[int(input.axis1()),int(input.axis2())],
                                      title=input.col_title(),
                                      color=input.col_text_color(),
                                      col_sup=col_sup,
                                      text_size=input.col_text_size(),
                                      lim_contrib = input.col_lim_contrib(),
                                      lim_cos2 = input.col_lim_cos2(),
                                      repel=input.col_plot_repel())
                return fig+pn.theme_gray()

            # Render Columns Plot
            @render.plot(alt="Columns Factor Map - CA")
            def fviz_col_plot():
                return plot_col().draw()
            
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
            
            #-------------------------------------------------------------------------------------------
            ## Eigenvalue informations
            #-------------------------------------------------------------------------------------------
            # Reactive Scree  plot
            @reactive.Calc
            def plot_eigen():
                return fviz_eig(self=model,choice=input.fviz_eigen_choice(),add_labels=input.fviz_eigen_label(),ggtheme=pn.theme_gray())

            # Render Scree plot
            @render.plot(alt="Scree Plot - CA")
            def fviz_eigen():
                return plot_eigen().draw()

            # Eigen value - DataFrame
            @render.data_frame
            def eigen_table():
                eig = model.eig_.round(4).reset_index().rename(columns={"index":"dimensions"})
                eig.columns = [x.capitalize() for x in eig.columns]
                return DataTable(data=match_datalength(eig,input.eigen_table_len()),filters=input.eigen_table_filter())

            #--------------------------------------------------------------------------------------------
            ##      Columns informations
            #---------------------------------------------------------------------------------------------
            # Factor coordinates
            @render.data_frame
            def col_coord_table():
                col_coord = model.col_["coord"].round(4).reset_index()
                col_coord.columns = ["Columns", *col_coord.columns[1:]]
                return DataTable(data=match_datalength(data=col_coord,value=input.col_coord_len()),filters=input.col_coord_filter())

            # Columns Contributions
            @render.data_frame
            def col_contrib_table():
                col_contrib = model.col_["contrib"].round(4).reset_index()
                col_contrib.columns = ["Columns", *col_contrib.columns[1:]]
                return  DataTable(data=match_datalength(data=col_contrib,value=input.col_contrib_len()),filters=input.col_contrib_filter())

            # Add Columns Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.col_contrib_graph_btn)
            def _():
                graph_modal_show(text="col",name="contrib",max_axis=model.call_["n_components"])

            # Reactive Columns Contributions Map
            @reactive.Calc
            def col_contrib_plot():
                fig = fviz_contrib(self=model,choice="col",axis=input.col_contrib_axis(),top_contrib=int(input.col_contrib_top()),color=input.col_contrib_color(),bar_width=input.col_contrib_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot columns Contributions
            @render.plot(alt="Columns Contributions Map - CA")
            def fviz_col_contrib():
                return col_contrib_plot().draw()

            # Square cosinus
            @render.data_frame
            def col_cos2_table():
                col_cos2 = model.col_["cos2"].round(4).reset_index()
                col_cos2.columns = ["Columns", *col_cos2.columns[1:]]
                return  DataTable(data=match_datalength(data=col_cos2,value=input.col_cos2_len()),filters=input.col_cos2_filter())

            # Add Columns Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.col_cos2_graph_btn)
            def _():
                graph_modal_show(text="col",name="cos2",max_axis=model.call_["n_components"])

            # Reactive Graph
            @reactive.Calc
            def col_cos2_plot():
                fig = fviz_cos2(self=model,choice="col",axis=input.col_cos2_axis(),top_cos2=int(input.col_cos2_top()),color=input.col_cos2_color(),bar_width=input.col_cos2_bar_width(),ggtheme=pn.theme_gray())
                return fig

            # Plot Columns Cos2
            @render.plot(alt="Columns Cosines Map - CA")
            def fviz_col_cos2():
                return col_cos2_plot().draw()

            #---------------------------------------------------------------------------------
            ## Supplementary Columns
            #------------------------------------------------------------------------------------
            if hasattr(model,"col_sup_"):
                @render.ui
                def col_sup_panel():
                    return ui.panel_conditional("input.value_choice == 'col_sup_res'",
                                ui.input_radio_buttons(id="col_sup_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                                ui.panel_conditional("input.col_sup_choice === 'coord'",panel_conditional1(text="col_sup",name="coord")),
                                ui.panel_conditional("input.col_sup_choice === 'cos2'",panel_conditional1(text="col_sup",name="cos2"))
                            )
            
                # Supplementary columns coordinates
                @render.data_frame
                def col_sup_coord_table():
                    col_sup_coord = model.col_sup_["coord"].round(4).reset_index()
                    col_sup_coord.columns = ["Columns", *col_sup_coord.columns[1:]]
                    return DataTable(data=match_datalength(data=col_sup_coord,value=input.col_sup_coord_len()),filters=input.col_sup_coord_filter())

                # Supplementary columns Cos2
                @render.data_frame
                def col_sup_cos2_table():
                    col_sup_cos2 = model.col_sup_["cos2"].round(4).reset_index()
                    col_sup_cos2.columns = ["Columns", *col_sup_cos2.columns[1:]]
                    return DataTable(data=match_datalength(data=col_sup_cos2,value=input.col_sup_cos2_len()),filters=input.col_sup_cos2_filter())
            
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
                
                # Factor coordinates
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
            
            #------------------------------------------------------------------------------------------
            ## Supplementary qualitative variables
            #-----------------------------------------------------------------------------------------
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

            #---------------------------------------------------------------------------------------------
            ## Row Points Informations
            #---------------------------------------------------------------------------------------------
            # Rows Coordinates
            @render.data_frame
            def row_coord_table():
                row_coord = model.row_["coord"].round(4).reset_index()
                row_coord.columns = ["Rows", *row_coord.columns[1:]]
                return DataTable(data = match_datalength(row_coord,input.row_coord_len()),filters=input.row_coord_filter())

            # Rows Contributions
            @render.data_frame
            def row_contrib_table():
                row_contrib = model.row_["contrib"].round(4).reset_index()
                row_contrib.columns = ["Rows", *row_contrib.columns[1:]]
                return  DataTable(data=match_datalength(row_contrib,input.row_contrib_len()),filters=input.row_contrib_filter())

            # Add rows Contributions Modal Show
            @reactive.Effect
            @reactive.event(input.row_contrib_graph_btn)
            def _():
                graph_modal_show(text="row",name="contrib",max_axis=model.call_["n_components"])

            # Plot Rows Contributions
            @reactive.Calc
            def row_contrib_plot():
                fig = fviz_contrib(self=model,choice="row",axis=input.row_contrib_axis(),top_contrib=int(input.row_contrib_top()),color = input.row_contrib_color(),bar_width= input.row_contrib_bar_width(),ggtheme=pn.theme_gray())
                return fig

            @render.plot(alt="Rows Contributions Map - CA")
            def fviz_row_contrib():
                return row_contrib_plot().draw()

            # Rows Cos2
            @render.data_frame
            def row_cos2_table():
                row_cos2 = model.row_["cos2"].round(4).reset_index()
                row_cos2.columns = ["Rows", *row_cos2.columns[1:]]
                return  DataTable(data = match_datalength(row_cos2,input.row_cos2_len()),filters=input.row_cos2_filter())

            # Add Rows Cos2 Modal Show
            @reactive.Effect
            @reactive.event(input.row_cos2_graph_btn)
            def _():
                graph_modal_show(text="row",name="cos2",max_axis=model.call_["n_components"])

            # Plot Rows Cos2
            @reactive.Calc
            def row_cos2_plot():
                fig = fviz_cos2(self=model,choice="row",axis=input.row_cos2_axis(),top_cos2=int(input.row_cos2_top()),color=input.row_cos2_color(),bar_width=input.row_cos2_bar_width(),ggtheme=pn.theme_gray())
                return fig

            @render.plot(alt="Rows Cosines Map - CA")
            def fviz_row_cos2():
                return row_cos2_plot().draw()
            
            #-------------------------------------------------------------------------------------------------
            ## Supplementary Rows informations
            #-------------------------------------------------------------------------------------------------
            if hasattr(model,"row_sup_"):
                @render.ui
                def row_sup_panel():
                    return ui.panel_conditional("input.value_choice == 'row_sup_res'",
                                ui.input_radio_buttons(id="row_sup_choice",label=ui.h6("Quel type de résultats?"),choices={"coord":"Coordonnées","cos2":"Cos2 - Qualité de la représentation"},selected="coord",width="100%",inline=True),
                                ui.panel_conditional("input.row_sup_choice === 'coord'",panel_conditional1(text="row_sup",name="coord")),
                                ui.panel_conditional("input.row_sup_choice === 'cos2'",panel_conditional1(text="row_sup",name="cos2"))
                            )

                # Factor coordinates
                @render.data_frame
                def row_sup_coord_table():
                    row_sup_coord = model.row_sup_["coord"].round(4).reset_index()
                    row_sup_coord.columns = ["Rows", *row_sup_coord.columns[1:]]
                    return  DataTable(data = match_datalength(row_sup_coord,input.row_sup_coord_len()),filters=input.row_sup_coord_filter())

                # Square cosinus
                @render.data_frame
                def row_sup_cos2_table():
                    row_sup_cos2 = model.row_sup_["cos2"].round(4).reset_index()
                    row_sup_cos2.columns = ["Rows", *row_sup_cos2.columns[1:]]
                    return  DataTable(data = match_datalength(row_sup_cos2,input.row_sup_cos2_len()),filters=input.row_sup_cos2_filter())

            #-------------------------------------------------------------------------------------------------
            ## Summary of data
            #-------------------------------------------------------------------------------------------------
            # Distribution conditionelle (X/Y)
            @render.data_frame
            def cond_dist_one_table():
                data = model.call_["X"].apply(lambda x : 100*x/np.sum(x),axis=0)
                data.loc["Total",:] = data.sum(axis=0)
                data = data.round(4).reset_index()
                data.columns = ["Rows", *data.columns[1:]]
                return DataTable(data = match_datalength(data,input.cond_dist_one_len()),filters=input.cond_dist_one_filter())

            # Distribution conditionelle (Y/X)
            @render.data_frame
            def cond_dist_two_table():
                data = model.call_["X"].apply(lambda x : 100*x/np.sum(x),axis=1)
                data.loc[:,"Total"] = data.sum(axis=1)
                data = data.round(4).reset_index()
                data.columns = ["Rows", *data.columns[1:]]
                return DataTable(data = match_datalength(data,input.cond_dist_two_len()),filters=input.cond_dist_two_filter())
            
            if hasattr(model,"quanti_sup_"):
                pass
            
            if hasattr(model,"quali_sup_"):
                quali_sup_labels = model.quali_sup_["eta2"].index.tolist()
                @render.ui
                def quali_sup_graph():
                    return ui.panel_conditional("input.resume_choice === 'bar_plot'",
                            ui.row(
                                ui.column(2,
                                    ui.input_select(id="quali_sup_label",label=ui.h6("Choisir une variable"),choices={x:x for x in quali_sup_labels},selected=quali_sup_labels[0])
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
                        )
                
                @reactive.Calc
                def plot_bar():
                    data = model.call_["Xtot"].loc[:,quali_sup_labels].astype("object")
                    if hasattr(model,"row_sup_"):
                        data = data.drop(index=model.call_["row_sup"])
                    return pn.ggplot(data,pn.aes(x=input.quali_sup_label()))+ pn.geom_bar(color="black",fill="gray")

                # Diagramme en barres
                @render.plot(alt="Bar-Plot")
                def fviz_bar_plot():
                    return plot_bar().draw()

            #---------------------------------------------------------------------------------------------------
            ## Overall Data
            #---------------------------------------------------------------------------------------------------
            @render.data_frame
            def overall_data_table():
                overall_data = model.call_["Xtot"].reset_index()
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