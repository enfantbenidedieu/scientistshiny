
###################################

from shiny import render, ui
import matplotlib.colors as mcolors
import plotnine as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fviz_barplot(X,
                 ncp = 1,
                 axis=None,
                 xlabel=None,
                 ylabel = None,
                 top_corr=10,
                 title = None,
                 bar_width=None,
                 add_grid=True,
                 color="steelblue",
                 ggtheme=pn.theme_gray()) -> plt:   
        
    if axis is None:
        axis = 0
    elif not isinstance(axis,int):
        raise ValueError("Error : 'axis' must be an integer.")
    elif axis < 0 or axis > ncp:
        raise ValueError(f"Error : 'axis' must be an integer between 0 and {ncp- 1}.")
            
    if xlabel is None:
        xlabel = ""
            
    if bar_width is None:
        bar_width = 0.5
    if top_corr is None:
        top_corr = 10
    elif not isinstance(top_corr,int):
        raise ValueError("Error : 'top_corr' must be an integer.")
    
    corr = X.iloc[:,axis].values
    labels = X.index
    
    n = len(labels)
    n_labels = len(labels)
        
    if (top_corr is not None) & (top_corr < n_labels):
        n_labels = top_corr
        
    limit = n - n_labels
    contrib_sorted = np.sort(corr)[limit:n]
    labels_sort = pd.Series(labels)[np.argsort(corr)][limit:n]

    df = pd.DataFrame({"labels" : labels_sort, "corr" : contrib_sorted})
    p = pn.ggplot(df,pn.aes(x = "reorder(labels,corr)", y = "corr"))+pn.geom_bar(stat="identity",fill=color,width=bar_width)

    if title is not None:
        p = p = p + pn.ggtitle(title)
    if ylabel is not None:
        p  = p + pn.xlab(ylabel)
    if xlabel is not None:
        p = p + pn.ylab(xlabel)

    # Coord Flip
    p = p + pn.coord_flip()

    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"),
                         axis_text_x = pn.element_text(angle = 90, ha = "center", va = "center"))

    return p+ggtheme


# Download Btn Background
download_btn_style = "background-color: #1C2951;"

# PanelContionnal without graph
def PanelConditional1(text=str,name=str):
    panel = ui.row(
                ui.column(2,
                    ui.h6("Paramètres"),
                    ui.input_radio_buttons(id=text+name+"Len",label="Taille d'affichage",choices={x:x for x in ["head","tail","all"]},selected="head",inline=True),
                    ui.input_switch(id=text+name+"Filter",label="Filtrer le tableau",value=False),
                ),
                ui.column(10,ui.div(ui.output_data_frame(id=text+name+"Table"),align="center"))
            )   
    return panel

# Panel Conditional With graph
def PanelConditional2(text=str,name=str):
    # Set Valeu name
    if name == "Contrib":
        value_name = "Contribution"
    elif name == "Cos2":
        value_name = "Cosinus"
    elif name == "Corr":
        value_name = "Correlation"
    elif name == "Vtest":
        value_name = "Vtest"

    panel = ui.row(
                ui.column(2,
                    ui.h6("Paramètres"),
                    ui.input_radio_buttons(id=text+name+"Len",label="Taille d'affichage",choices={x:x for x in ["head","tail","all"]},selected="head",inline=True),
                    ui.input_switch(id=text+name+"Filter",label="Filtrer le tableau",value=False),
                    ui.input_action_button(id=text+name+"GraphBtn",label="Graphe "+value_name,style = download_btn_style),
                    ui.input_action_button(id=text+name+"CorrGraphBtn",label="Graphe "+name+"-corr",style=download_btn_style)
                ),
                ui.column(10,ui.div(ui.output_data_frame(id=text+name+"Table"),align="center"))
            )   
    return panel

# Overall Panel Codnitionnal
def OverallPanelConditional(text):
    panel = ui.panel_conditional(f"input.choice == '{text}Res'",
                ui.br(),
                ui.h5("Coordonnées"),
                PanelConditional1(text=text,name="Coord"),
                ui.hr(),
                ui.h5("Contributions"),
                PanelConditional2(text=text,name="Contrib"),
                ui.hr(),
                ui.h5("Cos2 - Qualité de la représentation"),
                PanelConditional2(text=text,name="Cos2")
            )
    return panel

# Match with data
def match_datalength(data,value):
    match value:
        case "head":
            return data.head(6)
        case "tail":
            return data.tail(6)
        case "all":
            return data
        
# Return DaaFrame as DaaTable
def DataTable(data,filters=False):
    return render.DataTable(data,filters=filters,width="100%",row_selection_mode="multiple")

# 
def GraphModalShow(text=str,name=str):
    m = ui.modal(
            ui.output_plot(id=text+name+"Plot"),
            title=ui.div(
                ui.row(
                    ui.column(3,ui.input_numeric(id=text+name+"Axis",label="Choix de l'axe :",min=0,max=5,value=0)),
                    ui.column(3,ui.input_text(id=text+name+"Top",label="Top "+name,value=10,placeholder="Entrer un nombre")),
                    ui.column(3,ui.input_select(id=text+name+"Color",label="Couleur",choices={x:x for x in mcolors.CSS4_COLORS},selected="steelblue")),
                    ui.column(3,ui.input_slider(id=text+name+"BarWidth",label="Largeur des barres",min=0.1,max=1,value=0.5,step=0.1))
                ),
                class_="d-flex gap-4"
            ),
            easy_close=True,
            footer=ui.download_button(id=text+name+"GraphDownloadBtn",label="Download"),
            size="l"
        )
    return ui.modal_show(m)
    
# Modal Show
def GraphModelModal2(text=str,name=str,title=None):
    m = ui.modal(
            ui.output_plot(id=text+name+"CorrPlot"),
            title=ui.div(
                ui.row(
                    ui.column(3,ui.input_text(id=text+name+"CorrTitle",label="Titre du graphique",value=title,placeholder="Entrer un titre")),
                    ui.column(3,ui.input_select(id=text+name+"CorrColor",label="Couleur de bordure",choices={x:x for x in mcolors.CSS4_COLORS},selected="steelblue")),
                    ui.column(2,ui.input_select(id=text+name+"CorrLowColor",label="Low",choices={x:x for x in mcolors.CSS4_COLORS},selected="blue")),
                    ui.column(2,ui.input_select(id=text+name+"CorrMidColor",label="Medium",choices={x:x for x in mcolors.CSS4_COLORS},selected="white")),
                    ui.column(2,ui.input_select(id=text+name+"CorrHightColor",label="High",choices={x:x for x in mcolors.CSS4_COLORS},selected="red"))
                ),
                class_="d-flex gap-4"
            ),
            easy_close=True,
            footer=ui.download_button(id=text+name+"CorrGraphDownloadBtn",label="Download"),
            size="l"
        )
    return ui.modal_show(m)

