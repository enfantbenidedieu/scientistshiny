# -*- coding: utf-8 -*-
from shiny import ui, render
import matplotlib.colors as mcolors
from pathlib import Path

# CSS path
css_path = Path(__file__).parent / "www" / "style.css"

# Download Btn Background
download_btn_style = "background-color: #1C2951;"

def header(title=None,model_name=None):
    return ui.panel_title(ui.div(ui.h1(title),align="center",style="background-color:#2e4053;font-family: Cambria,Georgia,serif;"),window_title=model_name+"shiny")

def axes_input_select(model=None):
    return ui.TagList(
        ui.div(ui.input_select(id="axis1",label="",choices={x:x for x in range(model.call_["n_components"])},selected=0,multiple=False),style="display: inline-block;"),
        ui.div(ui.input_select(id="axis2",label="",choices={x:x for x in range(model.call_["n_components"])},selected=1,multiple=False),style="display: inline-block;"),
    )

def text_size_input(which=None):
    return ui.input_slider(id=which+"_text_size",label="Taille des libellés",min=8,max=20,value=8,step=2,ticks=False)

def text_color_input(id=None,choices=None):
    return ui.input_select(id=id,label="Colorier les points par :",choices=choices,selected="actif/sup",multiple=False,width="100%")

def graph_input(id=None,choices=None):
    return ui.input_radio_buttons(id=id,label="Modifier le graphe des",choices=choices,selected=list(choices.keys())[0],inline=True,width="100%")

def title_input(id=None,value=None):
    return ui.input_text(id=id,label="Titre du graphe",value=value,width="100%")

def point_select_input(id=None):
    return ui.input_select(id=id,label="Libellés des points pour",choices={"none":"Pas de sélection","cos2":"Cosinus","contrib" : "Contribution"},selected="none",multiple=False,width="100%")

def lim_cos2(id=None):
    return ui.input_slider(id=id,label = "Libellés pour un cos2 plus grand que",min = 0, max = 1,value=0,step=0.05)

def lim_contrib(id=None):
    return ui.input_slider(id=id,label ="Libellés pour une contribution plus grande que",min = 0, max = 100,value=0,step=5)

def dim_desc_panel(model=None):
    n_components = min(3,model.call_["n_components"])
    dim_desc_choice = {}
    for i in range(n_components):
        dim_desc_choice.update({"Dim."+str(i+1) : "Dimension "+str(i+1)})
    return ui.nav_panel("Description automatique des axes",
                ui.row(
                    ui.column(7,ui.input_radio_buttons(id="dim_desc_pvalue",label=ui.h6("Probabilité critique"),choices={x:y for x,y in zip([0.01,0.05,0.1,1.0],["Significance level 1%","Significance level 5%","Significance level 10%","None"])},selected=0.05,width="100%",inline=True)),
                    ui.column(5,ui.input_radio_buttons(id="dim_desc_axis",label=ui.h6("Choisir les dimensions"),choices=dim_desc_choice,selected="Dim.1",inline=True))
                ),
                ui.output_ui(id="dim_desc")
            )

def reset_columns(X=None):
    level0, level1 = X.columns.get_level_values(0), X.columns.get_level_values(1)
    X.columns = [str(x)+"."+str(y) for x, y in zip(level1,level0)]
    return X

def eigen_panel():
    return ui.panel_conditional("input.value_choice === 'eigen_res'",
                ui.input_radio_buttons(id="eigen_choice",label=ui.h6("Quel type de résultats?"),choices={"graph":"Graphes","table" : "Table"},selected="graph",width="100%",inline=True),
                ui.panel_conditional("input.eigen_choice === 'graph'",
                    ui.row(
                        ui.column(2,
                            ui.input_radio_buttons(id="fviz_eigen_choice",label=ui.h6("Choice"),choices={"eigenvalue" : "Eigenvalue","proportion" : "Proportion"},selected="proportion",width="100%",inline=False),
                            ui.div(ui.input_switch(id="fviz_eigen_label",label="Etiquettes",value=True),align="left")
                        ),
                        ui.column(10,
                            ui.div(ui.output_plot("fviz_eigen",width='100%', height='500px'),align="center"),
                            ui.hr(),
                            ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px"),
                            ui.div(ui.download_button(id="download_eigen_plot_jpg",label="jpg",style = download_btn_style),style="display: inline-block;"),
                            ui.div(ui.download_button(id="download_eigen_plot_png",label="png",style = download_btn_style),style="display: inline-block;"),
                            ui.div(ui.download_button(id="download_eigen_plot_pdf",label="pdf",style = download_btn_style),style="display: inline-block;"),
                            align="center"
                        )
                    )
                ),
                ui.panel_conditional("input.eigen_choice === 'table'",
                    ui.row(
                        ui.column(2,
                            ui.h6("Paramètres"),
                            ui.input_radio_buttons(id="eigen_table_len",label="Taille d'affichage",choices={x:x for x in ["head","tail","all"]},selected="all",inline=True),
                            ui.input_switch(id="eigen_table_filter",label="Filtrer le tableau",value=False),
                        ),
                        ui.column(10,
                            ui.div(ui.output_data_frame(id="eigen_table"),align="center"),
                            ui.hr(),
                            ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                            ui.div(ui.download_button(id="eigen_download_xlsx",label="xlsx",style = download_btn_style),style="display: inline-block;"),
                            ui.div(ui.download_button(id="eigen_download_csv",label="csv",style = download_btn_style),style="display: inline-block;"),
                            ui.div(ui.download_button(id="eigen_download_txt",label="txt",style = download_btn_style),style="display: inline-block;"),
                            align="center"
                        )
                    ) 
                )
            )

# PanelContionnal without graph
def panel_conditional1(text=str,name=str):   
    return ui.row(
                ui.column(2,
                    ui.h6("Paramètres"),
                    ui.input_radio_buttons(id=text+"_"+name+"_len",label="Taille d'affichage",choices={x:x for x in ["head","tail","all"]},selected="all",inline=True),
                    ui.input_switch(id=text+"_"+name+"_filter",label="Filtrer le tableau",value=False)
                ),
                ui.column(10,
                    ui.div(ui.output_data_frame(id=text+"_"+name+"_table"),align="center"),
                    ui.hr(),
                    ui.div(ui.h6("Téléchargement"),style="display: inline-block;padding: 5px",align="center"),
                    ui.div(ui.download_button(id=text+"_"+name+"_download_xlsx",label="xlsx",style = download_btn_style),style="display: inline-block;"),
                    ui.div(ui.download_button(id=text+"_"+name+"_download_csv",label="csv",style = download_btn_style),style="display: inline-block;"),
                    ui.div(ui.download_button(id=text+"_"+name+"_download_txt",label="txt",style = download_btn_style),style="display: inline-block;"),
                    align="center"
                )
            ) 

# Panel Conditional With button
def panel_conditional2(text=str,name=str):

    if name not in ["contrib","cos2"]:
        raise ValueError("'name' must be one of 'contrib','cos2'")
    # Set Valeu name
    if name == "contrib":
        value_name = "Contribution"
    elif name == "cos2":
        value_name = "Cosinus"
    
    return ui.row(
                ui.column(2,
                    ui.h6("Paramètres"),
                    ui.input_radio_buttons(id=text+"_"+name+"_len",label="Taille d'affichage",choices={x:x for x in ["head","tail","all"]},selected="all",inline=True),
                    ui.input_switch(id=text+"_"+name+"_filter",label="Filtrer le tableau",value=False),
                    ui.input_action_button(id=text+"_"+name+"_graph_btn",label="Graphe "+value_name,style = download_btn_style)
                ),
                ui.column(10,ui.div(ui.output_data_frame(id=text+"_"+name+"_table"),align="center"))
            )  

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

def graph_modal_show(text=str,name=str,max_axis=3):
    m = ui.modal(
            ui.output_plot(id="fviz_"+text+"_"+name),
            title=ui.div(
                ui.row(
                    ui.column(3,ui.input_numeric(id=text+"_"+name+"_axis",label="Choix de l'axe :",min=0,max=max_axis-1,value=0)),
                    ui.column(3,ui.input_text(id=text+"_"+name+"_top",label="Top "+name,value=10,placeholder="Entrer un nombre")),
                    ui.column(3,ui.input_select(id=text+"_"+name+"_color",label="Couleur",choices={x:x for x in mcolors.CSS4_COLORS},selected="steelblue")),
                    ui.column(3,ui.input_slider(id=text+"_"+name+"_bar_width",label="Largeur des barres",min=0.1,max=1,value=0.5,step=0.1))
                ),
                class_="d-flex gap-4"
            ),
            easy_close=True,
            footer=ui.download_button(id=text+"_"+name+"_graph_download_btn",label="Download"),
            size="xl"
        )
    return ui.modal_show(m)