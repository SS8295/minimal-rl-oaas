import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
import seaborn as sns

def save_df_as_image(df, path):
    # Set background to white
    norm = matplotlib.colors.Normalize(-1,1)
    colors = [[norm(-1.0), "white"],
            [norm( 1.0), "white"]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    # Make plot
    plot = sns.heatmap(df, annot=True, cmap=cmap, cbar=False)
    fig = plot.get_figure()
    fig.savefig(path)

def plot_learning_curve(scores, x, figure_file):

    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0,i-100):(i+1)])

    plt.plot(x,running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def render_mpl_table(data, col_width=2.0, row_height=0.625, font_size=10,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

    return ax.get_figure(), ax

def save_video():

    save_size = 0
    img_array = []
    for i in range(25):
        filename = './renders/live_render'+str(i)+'.png'
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        save_size = size
        img_array.append(img)
    
    
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2, save_size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def save_input_screenshot(state, sched_int):
    fig,ax = render_mpl_table(state[0], header_columns=0, col_width=3.0)
    fig.savefig("./tables/empl_"+str(sched_int)+"_input_.png")
    fig,ax = render_mpl_table(state[1], header_columns=0, col_width=3.0)
    fig.savefig("./tables/task_"+str(sched_int)+"_input_.png")  
    plt.close(fig)  

def save_output_screenshot(state, sched_int):
    fig,ax = render_mpl_table(state[0], header_columns=0, col_width=3.0)
    fig.savefig("./tables/empl_"+str(sched_int)+"_output_.png")
    fig,ax = render_mpl_table(state[1], header_columns=0, col_width=3.0)
    fig.savefig("./tables/task_"+str(sched_int)+"_output_.png")
    plt.close(fig)
