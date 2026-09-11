"""plot the main temporal response and linked orientation snapshots."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FuncFormatter


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "results/temporal/init_2x.json")
    parser.add_argument("--out-dir", type=Path, default=root / "outputs/temporal_figures")
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    main_results = json.loads(args.results.read_text())["runs"]
    seeds = ("8","9","10")
    ink, muted, spine = "#223B4C", "#788691", "#8D99A2"
    blue, orange, purple = "#286CA5", "#D48829", "#9565A3"
    plt.rcParams.update({
        "font.family":"DejaVu Sans", "font.size":11,
        "text.color":ink, "axes.labelcolor":ink, "axes.edgecolor":spine,
        "axes.labelsize":12, "axes.linewidth":.7, "axes.titlesize":13,
        "xtick.color":muted,"ytick.color":muted, "xtick.labelsize":10,"ytick.labelsize":10,
        "svg.fonttype":"none","pdf.fonttype":42,"savefig.facecolor":"white",
    })

    def rows(source, key):
        result = [source[key][seed] for seed in seeds]
        assert all(r["step"] == 24000 and r["window"] == "peak" for r in result)
        return result

    joint = rows(main_results,"fitted_joint")
    times = np.asarray(joint[0]["probe"]["times"])
    offsets = np.asarray(joint[0]["probe"]["offset_degrees"])
    visible = np.abs(offsets) <= 40
    center = offsets == 0
    flanks = (np.abs(offsets) >= 15) & (np.abs(offsets) <= 30)

    def arrays(records):
        return (np.asarray([r["probe"]["expected"] for r in records]),
                np.asarray([r["probe"]["baseline"] for r in records]))

    def response_change(records, mask):
        e,b = arrays(records)
        return 100*(e[:,:,mask].mean(-1)/b[:,:,mask].mean(-1)-1)

    def band(ax,x,y,color,ls="-",lw=2.3,alpha=.16,zorder=3):
        ax.fill_between(x,y.min(0),y.max(0),color=color,alpha=alpha,lw=0,zorder=zorder-1)
        return ax.plot(x,y.mean(0),color=color,lw=lw,ls=ls,zorder=zorder)[0]

    def style(ax, xticks, yticks, xlabel=None, ylabel=None, show_y=True):
        ax.spines[["top","right"]].set_visible(False)
        ax.spines["bottom"].set_position(("outward",5))
        ax.spines["left"].set_position(("outward",5))
        ax.spines["bottom"].set_bounds(xticks[0],xticks[-1])
        ax.spines["left"].set_bounds(yticks[0],yticks[-1])
        ax.set_xticks(xticks); ax.set_yticks(yticks)
        ax.tick_params(length=3.5,width=.65,pad=6)
        if xlabel: ax.set_xlabel(xlabel,labelpad=11)
        if ylabel: ax.set_ylabel(ylabel,labelpad=10)
        if not show_y:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y",left=False,labelleft=False)

    def letter(ax, text, x=-.105, y=1.075):
        ax.text(x,y,text,transform=ax.transAxes,fontsize=23,weight="bold",ha="left",va="bottom")

    def save(fig,stem):
        for ext in ("png","svg","pdf"):
            fig.savefig(out/f"{stem}.{ext}",dpi=230,facecolor="white")
        plt.close(fig)
        print(stem,flush=True)

    # Response snapshots linked to their actual locations in the full timecourse.
    fig=plt.figure(figsize=(14.8,9.4),facecolor="white")
    fig.text(.075,.955,"temporal response",fontsize=20,weight="bold",ha="left")
    fig.text(.075,.918,"expected stimulus · peak decoding",fontsize=11.5,color=muted)
    snap_axes=[fig.add_axes([.075+i*.235,.585,.205,.27]) for i in range(4)]
    trace_ax=fig.add_axes([.075,.10,.91,.345])
    e,b=arrays(joint)
    snapshot_times=(.1,1.,2.,4.)
    snap_indices=[int(np.argmin(abs(times-t))) for t in snapshot_times]
    for n,(ax,index,t) in enumerate(zip(snap_axes,snap_indices,snapshot_times)):
        for lo,hi in ((-30,-15),(15,30)):
            ax.axvspan(lo,hi,color=orange,alpha=.065,lw=0,zorder=0)
        band(ax,offsets[visible],b[:,index,visible],spine,ls=(0,(4,3)),lw=1.6,alpha=0)
        band(ax,offsets[visible],e[:,index,visible],ink,lw=2.3)
        ax.scatter([0],[e[:,index,center].mean()],s=27,color=blue,edgecolors="white",lw=.7,zorder=5)
        ax.scatter(offsets[flanks],e[:,index,flanks].mean(0),s=13,color=orange,edgecolors="white",lw=.35,zorder=5)
        ax.set_xlim(-40,40); ax.set_ylim(0,2.85)
        style(ax,[-30,0,30],[0,1,2],ylabel="rate (a.u.)" if n==0 else None,show_y=n==0)
        ax.set_xticklabels(["−30°","0°","+30°"])
        ax.set_title(f"t = {t:.1f}",weight="bold",pad=14)
        letter(ax,chr(97+n),x=-.13,y=1.06)
        connector=ConnectionPatch(xyA=(0,-.21),coordsA=ax.get_xaxis_transform(),
            xyB=(t,1.01),coordsB=trace_ax.get_xaxis_transform(),
            color="#B1BFCA",lw=.9,clip_on=False,zorder=0)
        fig.add_artist(connector)
        trace_ax.axvline(t,color="#C5D0D8",lw=.85,ls=(0,(3,4)),zorder=0)
    fig.text(.54,.512,"orientation offset",ha="center",color=muted,fontsize=11)
    fig.legend([Line2D([0],[0],color=ink,lw=2.3),
                Line2D([0],[0],color=spine,lw=1.6,ls=(0,(4,3)))],
               ["expected","baseline"],loc="upper right",bbox_to_anchor=(.985,.948),
               frameon=False,ncol=2,fontsize=11,handlelength=2.2,columnspacing=1.8)

    # Explicitly scaled late inset; retains the small central bump and off-center maxima.
    late_ax=snap_axes[-1].inset_axes([.53,.52,.45,.42])
    index=snap_indices[-1]
    band(late_ax,offsets[visible],e[:,index,visible],ink,lw=1.5)
    late_ax.set_xlim(-35,35); late_ax.set_ylim(0,.22)
    style(late_ax,[-30,0,30],[0,.1,.2])
    late_ax.tick_params(labelsize=7,length=2,pad=2)
    late_ax.set_xticklabels(["−30","0","30"])
    late_ax.set_title("zoom",fontsize=8.5,pad=5)
    late_ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f"{v:g}"))

    for mask,color in ((center,blue),(flanks,orange)):
        values=response_change(joint,mask)
        band(trace_ax,times,values,color)
        trace_ax.scatter(times[snap_indices],values.mean(0)[snap_indices],s=33,
                         color=color,edgecolors="white",linewidths=.9,zorder=5)
    trace_ax.axhline(0,color=spine,ls=(0,(4,3)),lw=.85,zorder=1)
    trace_ax.set_xlim(-.03,4.06); trace_ax.set_ylim(-100,105)
    style(trace_ax,[0,1,2,3,4],[-100,-50,0,50,100],
          xlabel="time (relative units)",ylabel="change from baseline (%)")
    trace_ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f"+{v:.0f}" if v>0 else f"{v:.0f}"))
    letter(trace_ax,"e",x=-.06,y=1.07)
    trace_ax.legend([
        Line2D([0],[0],color=blue,lw=2.3),
        Line2D([0],[0],color=orange,lw=2.3)],
        ["center: 0°","flanks: ±15–30°"],
        loc="center right",bbox_to_anchor=(.995,.65),ncol=2,frameon=False,
        fontsize=10.5,handlelength=2.4,columnspacing=2,labelspacing=.9)
    fig.text(.985,.018,"3 seeds · mean and range",ha="right",fontsize=9.5,color=muted)
    save(fig,"temporal_response")



if __name__ == "__main__":
    main()
