# -*- coding: utf-8 -*-
from shiny import render, ui
import matplotlib.colors as mcolors
import plotnine as pn

def fviz_barplot(X,
                 ncp = 2,
                 axis=None,
                 y_label = None,
                 top_corr=10,
                 title = None,
                 bar_width=None,
                 add_grid=True,
                 color="steelblue",
                 xtickslab_rotation = 45,
                 ggtheme=pn.theme_gray()) -> pn: 
    """
    
    
    
    """  
        
    if axis is None:
        axis = 0
    elif not isinstance(axis,int):
        raise ValueError("'axis' must be an integer.")
    elif axis < 0 or axis > ncp:
        raise ValueError(f"'axis' must be an integer between 0 and {ncp- 1}.")
            
    if bar_width is None:
        bar_width = 0.5
    if top_corr is None:
        top_corr = 10
    elif not isinstance(top_corr,int):
        raise ValueError("'top_corr' must be an integer.")
    
    #########
    corr = X.iloc[:,axis].reset_index()
    corr.columns = ["name","corr"]

    if top_corr is not None:
        corr = corr.sort_values(by="corr",ascending=False).head(top_corr)
    
    p = pn.ggplot()
    
    p = p + pn.geom_bar(data=corr,mapping=pn.aes(x="reorder(name,-corr)",y="corr",group = 1),
                        fill=color,color=color,width=bar_width,stat="identity")
    
    if y_label is None:
        y_label = "Cos2 - Quality of representation"
    p = p + pn.labs(title=title,y=y_label,x="")

    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))
    p = p + ggtheme

    if xtickslab_rotation > 5:
        ha = "right"
    if xtickslab_rotation == 90:
        ha = "center"

    # Rotation
    p = p + pn.theme(axis_text_x = pn.element_text(rotation = xtickslab_rotation,ha=ha))
   
    return p


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

    if name not in ["Contrib","Cos2"]:
        raise ValueError("'name' must be one of 'Contrib','Cos2'")
    # Set Valeu name
    if name == "Contrib":
        value_name = "Contribution"
    elif name == "Cos2":
        value_name = "Cosinus"
    
    panel = ui.row(
                ui.column(2,
                    ui.h6("Paramètres"),
                    ui.input_radio_buttons(id=text+name+"Len",label="Taille d'affichage",choices={x:x for x in ["head","tail","all"]},selected="head",inline=True),
                    ui.input_switch(id=text+name+"Filter",label="Filtrer le tableau",value=False),
                    ui.input_action_button(id=text+name+"GraphBtn",label="Graphe "+value_name,style = download_btn_style)
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
    return render.DataTable(data,filters=filters,selection_mode="rows")

# 
def GraphModalShow(text=str,name=str,max_axis=3):
    m = ui.modal(
            ui.output_plot(id=text+name+"Plot"),
            title=ui.div(
                ui.row(
                    ui.column(3,ui.input_numeric(id=text+name+"Axis",label="Choix de l'axe :",min=0,max=max_axis-1,value=0)),
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