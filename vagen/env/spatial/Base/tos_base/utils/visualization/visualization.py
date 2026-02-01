# visualization.py
import json
import os
from pathlib import Path
from html import escape
from typing import List, Dict, Optional
from .html_templates import HTML_TEMPLATE, CSS_STYLES, JAVASCRIPT_CODE

from ..utils import hash as stable_hash
from .charts import create_infogain_plot, create_cogmap_metrics_plot, create_correlation_plot, create_scalar_metric_plot
from ..cogmap.analysis import avg_nested_dicts, avg_float_list_skip_none



class VisualizationHelper:
    """Helper class for data processing and HTML generation"""
    
    @staticmethod
    def dict_to_html(d: Dict) -> str:
        """Convert dictionary to HTML format with better styling"""
        if not d:
            return "<div class='empty-dict'>(none)</div>"
        
        html = "<div class='dict-container'>"
        for k, v in d.items():
            if isinstance(v, (int, float)):
                # Format numbers nicely
                if isinstance(v, float):
                    if v != v:  # NaN check
                        formatted_v = "NaN"
                    else:
                        formatted_v = f"{v:.3f}" if v != int(v) else str(int(v))
                else:
                    formatted_v = str(v)
                html += f"<div class='dict-item'><span class='dict-key'>{escape(str(k))}:</span> <span class='dict-value number'>{formatted_v}</span></div>"
            elif isinstance(v, bool):
                # Color-code booleans
                color_class = "true" if v else "false"
                html += f"<div class='dict-item'><span class='dict-key'>{escape(str(k))}:</span> <span class='dict-value {color_class}'>{str(v)}</span></div>"
            elif isinstance(v, dict):
                # Handle nested dictionaries
                nested_html = VisualizationHelper.dict_to_html(v)
                html += f"<div class='dict-item nested'><span class='dict-key'>{escape(str(k))}:</span> <div class='dict-value nested-dict'>{nested_html}</div></div>"
            else:
                # String values with consistent styling
                html += f"<div class='dict-item'><span class='dict-key'>{escape(str(k))}:</span> <span class='dict-value string'>{escape(str(v))}</span></div>"
        html += "</div>"
        return html
    
    








class HTMLGenerator:
    """Handles HTML generation for the visualization"""
    
    def __init__(self, data: Dict, output_html: str, show_images: bool = True):
        self.data = data
        self.output_html = output_html
        self.show_images = show_images
        self.out_dir = os.path.abspath(os.path.dirname(output_html))
        self.html_dir = self.out_dir
        self.base = Path(output_html).stem
        self.page_urls: List[str] = []
        self.current_page: int = 0

        # Extract data
        self.meta = data.get("meta_info", {})
        self.samples = data.get("samples", {})
        self.total_samples = len(self.samples)

        # Extract summary data
        self.exp_summary = data.get("exp_summary", {})
        self.eval_summary = data.get("eval_summary", {})
        self.cogmap_summary = data.get("cogmap_summary", {})
        self.correlation_summary = data.get("correlation", {})

        # Calculate statistics - each sample becomes one page
        self.total_pages = 1 + self.total_samples  # page 0 = TOC

        # Build flat list for samples (sample_id, sample_data)
        self.flat = []
        for sample_id, sample_data in self.samples.items():
            self.flat.append((sample_id, sample_data))

        # Extract available combinations from sample data keys
        self.combinations = self._extract_combinations_from_samples()

    def _is_passive_combo(self, entry: Dict) -> bool:
        cfg = (entry or {}).get("config") or {}
        obs_cfg = cfg.get("observation_config") or {}
        return str(obs_cfg.get("exp_type", "")).lower() == "passive"

    def _relpath_from_html(self, p: str) -> str:
        if not isinstance(p, str) or not p:
            return p
        if os.path.isabs(p):
            if os.path.exists(p):
                return os.path.relpath(p, self.html_dir)
            return p

        candidates = [
            os.path.abspath(os.path.join(self.out_dir, p)),
            os.path.abspath(os.path.join(os.path.dirname(self.html_dir), p)),
            os.path.abspath(os.path.join(self.html_dir, p)),
        ]
        for ap in candidates:
            if os.path.exists(ap):
                return os.path.relpath(ap, self.html_dir)
        return p

    def _load_passive_prompt_context(self, entry: Dict) -> Optional[Dict[str, object]]:
        """Load (system/user) prompt and attached images before evaluation questions for passive runs."""
        if not self._is_passive_combo(entry):
            return None
        cfg = (entry or {}).get("config") or {}
        obs_cfg = cfg.get("observation_config") or {}
        room_dict = cfg.get("room_dict") or {}
        agent_dict = cfg.get("agent_dict") or {}
        if not room_dict or not agent_dict:
            return None

        a = dict(agent_dict)
        a.pop("pos", None)
        a.pop("ori", None)
        room_key = stable_hash(json.dumps({**room_dict, **a}, sort_keys=True))

        render_mode = str(obs_cfg.get("render_mode", ""))
        think_str = "think" if bool((obs_cfg.get("prompt_config") or {}).get("enable_think", False)) else "nothink"
        proxy_agent = str(obs_cfg.get("proxy_agent") or "")
        combo_dir = os.path.join(self.out_dir, room_key, render_mode, "passive", think_str, proxy_agent)
        msg_path = os.path.join(combo_dir, "messages.json")
        if not os.path.exists(msg_path):
            return None

        try:
            with open(msg_path, "r") as f:
                messages = json.load(f) or []
        except Exception:
            return None

        sys_prompt = ""
        user_prompt = ""
        images: List[str] = []
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            sys_prompt = messages[0].get("content") or ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                user_prompt = m.get("content") or ""
                images = list(m.get("images") or [])
                break

        # Resolve image paths: paths in messages.json are relative to project root (cwd)
        resolved_images = []
        for p in images:
            if not isinstance(p, str) or not p:
                continue
            # paths are stored relative to project root (cwd)
            abs_p = os.path.abspath(p)
            if os.path.exists(abs_p):
                resolved_images.append(os.path.relpath(abs_p, self.html_dir))
            else:
                # Fallback to existing logic
                resolved_images.append(self._relpath_from_html(p))
        return {"system": sys_prompt, "user": user_prompt, "images": resolved_images}

    def _extract_combinations_from_samples(self) -> List[str]:
        """Extract unique combination keys from all samples"""
        combination_keys = set()

        for sample_data in self.samples.values():
            for key in sample_data.keys():
                combination_keys.add(key)

        # Return as sorted list for consistent ordering
        return sorted(list(combination_keys))

    @staticmethod
    def _safe_slug(s: object) -> str:
        text = str(s or "")
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text).strip("_")

    def _get_sample_subdir(self, sample_id: str, sample_data: Dict) -> str:
        for entry in (sample_data or {}).values():
            if isinstance(entry, dict):
                subdir = (entry.get("config") or {}).get("_sample_subdir")
                if isinstance(subdir, str) and subdir:
                    return subdir
        safe = self._safe_slug(sample_id)
        return safe or stable_hash(str(sample_id))

    @staticmethod
    def _format_eval_task_label(mode_label: str, task_type: str) -> str:
        if mode_label == "default":
            return task_type
        return f"{mode_label} / {task_type}"

    @staticmethod
    def _order_fb_metrics(metrics: Dict) -> Dict:
        if not isinstance(metrics, dict):
            return metrics
        preferred = ["inertia", "changed", "retention", "unchanged", "unchanged_retention", "unchanged_retention_minus_retention", "unchanged_exploration"]
        ordered = {k: metrics[k] for k in preferred if k in metrics}
        for k, v in metrics.items():
            if k not in ordered and k != "inertia_list":  # exclude inertia_list from display
                ordered[k] = v
        return ordered

    def generate_config_summaries(self, f) -> None:
        """Generate summaries for each config combination"""
        f.write("<div class='config-summaries'>\n")
        f.write("<h3>📋 Configuration Summaries</h3>\n")

        # Create plots directory for summary plots
        plots_dir = os.path.join(self.out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        for gname in self.combinations:
            # Count samples that have this combination
            sample_count = sum(1 for sample_data in self.samples.values()
                             if gname in sample_data and sample_data[gname] is not None)
            f.write(f"<div class='config-summary'>\n")
            f.write(f"<h4>⚙️ {escape(gname)}</h4>\n")
            f.write(f"<div class='config-stats'>\n")
            f.write(f"<div class='stat-item'>📊 Samples: {sample_count}</div>\n")
            f.write("</div>\n")

            # Generate plot data first (to pop values before metrics display)
            infogain_plot = None
            cogmap_update_plot = None
            cogmap_full_plot = None
            cogmap_self_tracking_plot = None
            cogmap_fb_unchanged_plot = None
            consistency_plots = {}


            # Exploration infogain plot
            if self.exp_summary.get("group_performance", {}).get(gname):
                exp_group = self.exp_summary["group_performance"][gname]
                infogain_per_turn = exp_group.pop("infogain_per_turn", [])
                if infogain_per_turn:
                    save_path = os.path.join(plots_dir, f"infogain_{gname}.png")
                    infogain_plot = create_infogain_plot(infogain_per_turn, gname, save_path, base_dir=self.out_dir)

            # Cognitive map plots (only global now)
            if self.cogmap_summary.get("group_performance", {}).get(gname):
                cogmap_group = self.cogmap_summary["group_performance"][gname]
                per_turn = cogmap_group.pop("per_turn_metrics", {}) if isinstance(cogmap_group, dict) else {}
                update_data = per_turn.get("cogmap_update_per_turn", cogmap_group.pop("cogmap_update_per_turn", {}))
                full_data = per_turn.get("cogmap_full_per_turn", cogmap_group.pop("cogmap_full_per_turn", {}))
                self_tracking_data = per_turn.get("self_tracking_per_turn", cogmap_group.pop("self_tracking_per_turn", {}))
                fb_unchanged_series = per_turn.get("cogmap_fb_unchanged_per_turn", cogmap_group.pop("cogmap_fb_unchanged_per_turn", {}))
                fog_probe_f1 = per_turn.get("fog_probe_f1_per_turn", cogmap_group.pop("fog_probe_f1_per_turn", []))
                fog_probe_p = per_turn.get("fog_probe_p_per_turn", cogmap_group.pop("fog_probe_p_per_turn", []))
                fog_probe_r = per_turn.get("fog_probe_r_per_turn", cogmap_group.pop("fog_probe_r_per_turn", []))
                
                # New consistency metrics
                pos_up = per_turn.get("position_update_per_turn", cogmap_group.pop("position_update_per_turn", []))
                fac_up = per_turn.get("facing_update_per_turn", cogmap_group.pop("facing_update_per_turn", []))
                pos_stab = per_turn.get(
                    "position_stability_per_turn",
                    per_turn.get("stability_per_turn", cogmap_group.pop("stability_per_turn", [])),
                )
                fac_stab = per_turn.get("facing_stability_per_turn", cogmap_group.pop("facing_stability_per_turn", []))

                # Only accept new shape (metric -> list)
                global_update = update_data if isinstance(update_data, dict) else {}
                if global_update and any(global_update.values()):
                    title = f"{gname} - Global (Update)"
                    save_path = os.path.join(plots_dir, f"cogmap_update_{gname}.png")
                    cogmap_update_plot = create_cogmap_metrics_plot(global_update, title, save_path=save_path, base_dir=self.out_dir)

                # Full mode plot (global only)
                global_full = full_data if isinstance(full_data, dict) else {}
                if global_full and any(global_full.values()):
                    title = f"{gname} - Global (Full)"
                    save_path = os.path.join(plots_dir, f"cogmap_full_{gname}.png")
                    cogmap_full_plot = create_cogmap_metrics_plot(global_full, title, save_path=save_path, base_dir=self.out_dir)

                # Self-tracking plot (agent only)
                global_self_tracking = self_tracking_data if isinstance(self_tracking_data, dict) else {}
                if global_self_tracking and any(global_self_tracking.values()):
                    title = f"{gname} - Global (Self-Tracking)"
                    save_path = os.path.join(plots_dir, f"cogmap_self_tracking_{gname}.png")
                    cogmap_self_tracking_plot = create_cogmap_metrics_plot(global_self_tracking, title, save_path=save_path, base_dir=self.out_dir)

                # Fog Probe plots
                fog_probe_plots = {}
                if isinstance(fog_probe_f1, list):
                    save_path = os.path.join(plots_dir, f"fog_probe_f1_{gname}.png")
                    fog_probe_plots['f1'] = create_scalar_metric_plot(fog_probe_f1, title=f"Fog Probe F1 per Turn - {gname}", y_label="F1", ylim=(0.0, 1.0), save_path=save_path, base_dir=self.out_dir)
                if isinstance(fog_probe_p, list):
                    save_path = os.path.join(plots_dir, f"fog_probe_p_{gname}.png")
                    fog_probe_plots['p'] = create_scalar_metric_plot(fog_probe_p, title=f"Fog Probe Precision per Turn - {gname}", y_label="Precision", ylim=(0.0, 1.0), save_path=save_path, base_dir=self.out_dir)
                if isinstance(fog_probe_r, list):
                    save_path = os.path.join(plots_dir, f"fog_probe_r_{gname}.png")
                    fog_probe_plots['r'] = create_scalar_metric_plot(fog_probe_r, title=f"Fog Probe Recall per Turn - {gname}", y_label="Recall", ylim=(0.0, 1.0), save_path=save_path, base_dir=self.out_dir)

                fb_series = fb_unchanged_series if isinstance(fb_unchanged_series, dict) else {}
                if fb_series and any(fb_series.values()):
                    title = f"{gname} - False Belief (Unchanged)"
                    save_path = os.path.join(plots_dir, f"cogmap_fb_unchanged_{gname}.png")
                    cogmap_fb_unchanged_plot = create_cogmap_metrics_plot(fb_series, title, save_path=save_path, base_dir=self.out_dir)

            if isinstance(pos_up, list):
                save_path = os.path.join(plots_dir, f"position_update_{gname}.png")
                consistency_plots['pos_up'] = create_scalar_metric_plot(pos_up, title=f"Position Update - {gname}", y_label="Score", ylim=(0.0, 1.0), save_path=save_path, base_dir=self.out_dir)
            if isinstance(fac_up, list):
                save_path = os.path.join(plots_dir, f"facing_update_{gname}.png")
                consistency_plots['fac_up'] = create_scalar_metric_plot(fac_up, title=f"Facing Update - {gname}", y_label="Score", ylim=(0.0, 1.0), save_path=save_path, base_dir=self.out_dir)
            if isinstance(pos_stab, list):
                save_path = os.path.join(plots_dir, f"position_stability_{gname}.png")
                consistency_plots['pos_stab'] = create_scalar_metric_plot(pos_stab, title=f"Position Stability - {gname}", y_label="Score", ylim=(0.0, 1.0), save_path=save_path, base_dir=self.out_dir)
            if isinstance(fac_stab, list):
                save_path = os.path.join(plots_dir, f"facing_stability_{gname}.png")
                consistency_plots['fac_stab'] = create_scalar_metric_plot(fac_stab, title=f"Facing Stability - {gname}", y_label="Score", ylim=(0.0, 1.0), save_path=save_path, base_dir=self.out_dir)


            # Generate correlation plots
            correlation_plots = {}
            if self.correlation_summary.get("group_performance", {}).get(gname):
                correlation_data = self.correlation_summary["group_performance"][gname]
                cogmap_values = correlation_data.pop('last_global_vs_gt_fulls', [])
                acc_values = correlation_data.pop('avg_acc_metrics', [])
                infogain_values = correlation_data.pop('last_infogains', [])

                # Call twice to generate two scatter plots using existing correlation_info
                if cogmap_values and acc_values:
                    acc_correlation = correlation_data.get('cogmap_acc_correlations', {}).get('avg_accuracy', None)
                    save_path = os.path.join(plots_dir, f"correlation_cogmap_vs_accuracy_{gname}.png")
                    correlation_plots['cogmap_vs_accuracy'] = create_correlation_plot(
                        cogmap_values, acc_values,
                        'Cognitive Map Score (Last Global vs GT Full)',
                        'Average Accuracy',
                        'Cognitive Map Score vs Average Accuracy',
                        acc_correlation,
                        save_path,
                        base_dir=self.out_dir
                    )

                if cogmap_values and infogain_values:
                    infogain_correlation = correlation_data.get('cogmap_infogain_correlation', None)
                    save_path = os.path.join(plots_dir, f"correlation_cogmap_vs_infogain_{gname}.png")
                    correlation_plots['cogmap_vs_infogain'] = create_correlation_plot(
                        cogmap_values, infogain_values,
                        'Cognitive Map Score (Last Global vs GT Full)',
                        'Information Gain',
                        'Cognitive Map Score vs Information Gain',
                        infogain_correlation,
                        save_path,
                        base_dir=self.out_dir
                    )

            # Config metrics section with four-column layout (display metrics first)
            f.write("<div class='metrics-section'>\n")
            f.write("<div class='metrics-grid four-columns'>\n")

            # Group exploration performance
            exp_group = self.exp_summary.get("group_performance", {}).get(gname)
            if exp_group:
                exp_group_filtered = {k: v for k, v in exp_group.items() if k != "infogain_per_turn"}
                if exp_group_filtered:
                    f.write("<div class='metrics-box exploration'>\n")
                    f.write("<h4>🔍 Exploration</h4>\n")
                    f.write(VisualizationHelper.dict_to_html(exp_group_filtered))
                    f.write("</div>\n")

            # Group evaluation performance (default + extra modes)
            eval_by_mode = self.eval_summary.get("group_performance_by_mode", {})
            if eval_by_mode:
                for mode in sorted(eval_by_mode.keys()):
                    eval_group = (eval_by_mode.get(mode) or {}).get(gname)
                    if not eval_group:
                        continue
                    title = "✅ Evaluation" if mode == "default" else f"✅ Evaluation ({mode})"
                    f.write("<div class='metrics-box evaluation'>\n")
                    f.write(f"<h4>{escape(title)}</h4>\n")
                    f.write(VisualizationHelper.dict_to_html(self._filter_eval_for_display(eval_group)))
                    f.write("</div>\n")
            else:
                eval_group = self.eval_summary.get("group_performance", {}).get(gname)
                if eval_group:
                    f.write("<div class='metrics-box evaluation'>\n")
                    f.write("<h4>✅ Evaluation</h4>\n")
                    f.write(VisualizationHelper.dict_to_html(self._filter_eval_for_display(eval_group)))
                    f.write("</div>\n")

            # Group cognitive map performance
            cogmap_group = self.cogmap_summary.get("group_performance", {}).get(gname)
            if cogmap_group:
                # Extract fog_probe data (now at top level due to previous change)
                fog_probe_data = cogmap_group.get('fog_probe')
                cogmap_fb_data = cogmap_group.get('cogmap_fb')

                # Display main cognitive map metrics (exclude per_turn data AND fog_probe/cogmap_fb if present as top key)
                main_metrics = {k: v for k, v in cogmap_group.items()
                               if k not in ["cogmap_update_per_turn", "cogmap_full_per_turn", "self_tracking_per_turn",
                                            "fog_probe_f1_per_turn", "per_turn_metrics", "fog_probe", "cogmap_fb"]}
                if main_metrics:
                    f.write("<div class='metrics-box cogmap'>\n")
                    f.write("<h4>🧠 Cognitive Map</h4>\n")
                    f.write(VisualizationHelper.dict_to_html(main_metrics))
                    f.write("</div>\n")
                
                # Display Fog Probe separately
                if fog_probe_data:
                    f.write("<div class='metrics-box fog-probe'>\n")
                    f.write("<h4>🌫️ Fog Probe</h4>\n")
                    f.write(VisualizationHelper.dict_to_html(fog_probe_data))
                    f.write("</div>\n")
                
                # Display CogMap FB separately
                if cogmap_fb_data and cogmap_fb_data.get('metrics'):
                    f.write("<div class='metrics-box cogmap-fb'>\n")
                    f.write("<h4>🧭 False Belief CogMap</h4>\n")
                    f.write(VisualizationHelper.dict_to_html(self._order_fb_metrics(cogmap_fb_data['metrics'])))
                    f.write("</div>\n")

            # Group correlation performance
            correlation_summary = getattr(self, 'correlation_summary', {})
            correlation_group = correlation_summary.get("group_performance", {}).get(gname)
            if correlation_group:
                f.write("<div class='metrics-box correlation'>\n")
                f.write("<h4>📈 Correlation</h4>\n")
                f.write(VisualizationHelper.dict_to_html(correlation_group))
                f.write("</div>\n")

            f.write("</div>\n")  # End metrics-grid
            f.write("</div>\n")  # End metrics-section

            # Plots section (display after metrics, but plots were generated earlier)
            f.write("<div class='plots-section'>\n")

            # Display plots in a single row (up to 6 plots now)
            available_plots = []
            if infogain_plot:
                available_plots.append(("Information Gain per Turn", infogain_plot, "Information Gain per Turn"))
            if cogmap_update_plot:
                available_plots.append(("Cognitive Map (Update)", cogmap_update_plot, "Cognitive Map Update Turn Averages"))
            if cogmap_full_plot:
                available_plots.append(("Cognitive Map (Full)", cogmap_full_plot, "Cognitive Map Full Turn Averages"))
            if cogmap_self_tracking_plot:
                available_plots.append(("Cognitive Map (Self-Tracking)", cogmap_self_tracking_plot, "Cognitive Map Self-Tracking Turn Averages"))
            
            # Fog Probe
            if fog_probe_plots.get('f1'):
                available_plots.append(("Fog Probe F1", fog_probe_plots['f1'], "Fog Probe F1 per Turn"))
            if fog_probe_plots.get('p'):
                available_plots.append(("Fog Probe Precision", fog_probe_plots['p'], "Fog Probe Precision per Turn"))
            if fog_probe_plots.get('r'):
                available_plots.append(("Fog Probe Recall", fog_probe_plots['r'], "Fog Probe Recall per Turn"))

            if cogmap_fb_unchanged_plot:
                available_plots.append(("FB CogMap (Unchanged)", cogmap_fb_unchanged_plot, "False Belief CogMap Unchanged per Turn"))
            
            if consistency_plots.get('pos_up'):
                available_plots.append(("Position Update", consistency_plots['pos_up'], "Position Update per Turn"))
            if consistency_plots.get('fac_up'):
                available_plots.append(("Facing Update", consistency_plots['fac_up'], "Facing Update per Turn"))
            if consistency_plots.get('pos_stab'):
                available_plots.append(("Position Stability", consistency_plots['pos_stab'], "Position Stability per Turn"))
            if consistency_plots.get('fac_stab'):
                available_plots.append(("Facing Stability", consistency_plots['fac_stab'], "Facing Stability per Turn"))

            # Add correlation plots
            if correlation_plots.get('cogmap_vs_accuracy'):
                available_plots.append(("CogMap vs Accuracy", correlation_plots['cogmap_vs_accuracy'], "Cognitive Map vs Accuracy Correlation"))
            if correlation_plots.get('cogmap_vs_infogain'):
                available_plots.append(("CogMap vs InfoGain", correlation_plots['cogmap_vs_infogain'], "Cognitive Map vs Information Gain Correlation"))

            if available_plots:
                f.write("<div class='plots-row'>")
                f.write("<h5>Performance Charts</h5>")
                # Use flexible grid that can handle more plots
                f.write("<div class='plots-grid'>")
                for title, plot_uri, alt_text in available_plots:
                    plot_uri = self._relpath_from_html(plot_uri)
                    f.write(f"<div class='plot-item'>")
                    f.write(f"<h6>{title}</h6>")
                    f.write(f"<img src='{plot_uri}' alt='{alt_text}' class='plot-image'>")
                    f.write("</div>")
                f.write("</div>")
                f.write("</div>")

            f.write("</div>\n")  # End plots-section

            f.write("</div>\n")  # End config-summary

        f.write("</div>\n")


    def generate_toc_page(self, f) -> None:
        """Generate table of contents page with summaries"""
        f.write("<section class='sample-page active' id='page0'>\n")
        f.write("<h2>📋 Dashboard Overview</h2>\n")

        self.generate_config_summaries(f)

        f.write("<h3>📖 Sample Navigation</h3>\n")
        f.write("<ul>\n")
        running_page = 1

        for sample_id, sample_data in self.samples.items():
            # Count available combinations for this sample
            available_combos = [combo for combo in self.combinations
                               if combo in sample_data and sample_data[combo] is not None]
            combo_count = len(available_combos)

            href = self.page_urls[running_page] if len(self.page_urls) > running_page else "#"
            f.write(
                f"<li>"
                f"<a href='{href}#p{running_page + 1}'>"
                f"{escape(sample_id)} ({combo_count} combinations)</a>"
                f"</li>\n"
            )
            running_page += 1
        f.write("</ul>\n</section>\n")

    def generate_sample_metrics(self, f, entry: Dict, sample_name: str) -> None:
        """Generate sample-level metrics visualization"""
        metrics = entry.get("metrics", {})
        if not metrics:
            return

        f.write("<div class='metrics-section'>\n")
        f.write("<h3>📊 Sample Metrics</h3>\n")

        # Create a three-column layout for exploration, evaluation, and cogmap metrics
        f.write("<div class='metrics-grid'>\n")

        # Helper function to filter out per_turn keys
        def filter_per_turn_keys(data):
            if not isinstance(data, dict):
                return data
            return {k: v for k, v in data.items() if ("per_turn" not in k and k != "per_turn_metrics")}

        # Exploration metrics
        exploration_metrics = metrics.get("exploration", {})
        if exploration_metrics:
            filtered_exploration = filter_per_turn_keys(exploration_metrics)
            if filtered_exploration:
                f.write("<div class='metrics-box exploration'>\n")
                f.write("<h4>🔍 Exploration</h4>\n")
                f.write(VisualizationHelper.dict_to_html(filtered_exploration))
                f.write("</div>\n")

        # Evaluation metrics (default + extra modes)
        eval_blocks = []
        default_eval = self._filter_eval_for_display(metrics.get("evaluation", {}))
        if default_eval:
            eval_blocks.append(("✅ Evaluation", default_eval))
        for k, v in metrics.items():
            if k.startswith("evaluation_") and k != "evaluation":
                mode = k.split("evaluation_", 1)[1] or "default"
                eval_blocks.append((f"✅ Evaluation ({mode})", self._filter_eval_for_display(v)))
        for title, eval_metrics in eval_blocks:
            filtered_evaluation = filter_per_turn_keys(eval_metrics)
            if filtered_evaluation:
                f.write("<div class='metrics-box evaluation'>\n")
                f.write(f"<h4>{escape(title)}</h4>\n")
                f.write(VisualizationHelper.dict_to_html(filtered_evaluation))
                f.write("</div>\n")

        # Cognitive map metrics
        cogmap_metrics = metrics.get("cogmap", {})
        if cogmap_metrics:
            filtered_cogmap = filter_per_turn_keys(cogmap_metrics)
            if filtered_cogmap:
                f.write("<div class='metrics-box cogmap'>\n")
                f.write("<h4>🧠 Cognitive Map</h4>\n")
                f.write(VisualizationHelper.dict_to_html(filtered_cogmap))
                f.write("</div>\n")
        
        # False Belief Cogmap metrics
        cogmap_fb_metrics = (metrics.get("cogmap") or {}).get("cogmap_fb", {}) if isinstance(metrics.get("cogmap"), dict) else {}
        if cogmap_fb_metrics:
            fb_avg = cogmap_fb_metrics.get("metrics", {})
            if fb_avg:
                f.write("<div class='metrics-box cogmap-fb'>\n")
                f.write("<h4>🧭 False Belief CogMap</h4>\n")
                f.write(VisualizationHelper.dict_to_html(self._order_fb_metrics(fb_avg)))
                f.write("</div>\n")

        f.write("</div>\n")  # End metrics-grid
        f.write("</div>\n")  # End metrics-section

    def generate_cognitive_map_charts(self, f, entry: Dict, sample_name: str, sample_id: str) -> None:
        """Generate cognitive map charts and information gain chart in a single row"""
        # Create plots directory for this sample
        sample_subdir = entry.get("config", {}).get("_sample_subdir")
        if not sample_subdir:
            sample_subdir = self._get_sample_subdir(sample_id, {"": entry})
        plots_dir = os.path.join(self.out_dir, sample_subdir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Extract information gain data from exploration turns
        infogain_per_turn = entry['metrics'].get('exploration', {}).pop('infogain_per_turn', [])
        cogmap_metrics = entry['metrics'].get('cogmap', {}) or {}
        per_turn_metrics = cogmap_metrics.get('per_turn_metrics', {}) if isinstance(cogmap_metrics, dict) else {}
        cogmap_update_data = per_turn_metrics.get('cogmap_update_per_turn', None)
        cogmap_full_data = per_turn_metrics.get('cogmap_full_per_turn', None)
        self_tracking_data = per_turn_metrics.get('self_tracking_per_turn', None)
        fb_unchanged_per_turn = per_turn_metrics.get('cogmap_fb_unchanged_per_turn', None)
        fog_probe_f1_per_turn = per_turn_metrics.get('fog_probe_f1_per_turn', None)
        fog_probe_p_per_turn = per_turn_metrics.get('fog_probe_p_per_turn', None)
        fog_probe_r_per_turn = per_turn_metrics.get('fog_probe_r_per_turn', None)
        pos_up_per_turn = per_turn_metrics.get('position_update_per_turn', None)
        fac_up_per_turn = per_turn_metrics.get('facing_update_per_turn', None)
        pos_stab_per_turn = per_turn_metrics.get('position_stability_per_turn', None)
        if pos_stab_per_turn is None:
            pos_stab_per_turn = per_turn_metrics.get('stability_per_turn', None)  # backward compat
        fac_stab_per_turn = per_turn_metrics.get('facing_stability_per_turn', None)

        # Backward compatibility (older metric shape)
        if cogmap_update_data is None:
            cogmap_update_data = entry['metrics'].get('cogmap', {}).pop('cogmap_update_per_turn', {})
        if cogmap_full_data is None:
            cogmap_full_data = entry['metrics'].get('cogmap', {}).pop('cogmap_full_per_turn', {})
        if self_tracking_data is None:
            self_tracking_data = entry['metrics'].get('cogmap', {}).pop('self_tracking_per_turn', {})
        if fog_probe_f1_per_turn is None:
            fog_probe_f1_per_turn = entry['metrics'].get('cogmap', {}).pop('fog_probe_f1_per_turn', [])

        # Generate plots
        infogain_plot = None
        update_plot = None
        full_plot = None
        self_tracking_plot = None
        fb_unchanged_plot = None
        fog_probe_plots = {}

        # Information gain plot
        if infogain_per_turn:
            save_path = os.path.join(plots_dir, "infogain.png")
            infogain_plot = create_infogain_plot(infogain_per_turn, sample_name, save_path, base_dir=self.out_dir)

        # Cognitive map plots
        if any(cogmap_update_data.values()):
            title = f"{sample_name} - Global (Update)"
            save_path = os.path.join(plots_dir, "cogmap_update.png")
            update_plot = create_cogmap_metrics_plot(cogmap_update_data, title, save_path=save_path, base_dir=self.out_dir)

        if any(cogmap_full_data.values()):
            title = f"{sample_name} - Global (Full)"
            save_path = os.path.join(plots_dir, "cogmap_full.png")
            full_plot = create_cogmap_metrics_plot(cogmap_full_data, title, save_path=save_path, base_dir=self.out_dir)

        if any(self_tracking_data.values()):
            title = f"{sample_name} - Global (Self-Tracking)"
            save_path = os.path.join(plots_dir, "cogmap_self_tracking.png")
            self_tracking_plot = create_cogmap_metrics_plot(self_tracking_data, title, save_path=save_path, base_dir=self.out_dir)

        if isinstance(fb_unchanged_per_turn, dict) and any(fb_unchanged_per_turn.values()):
            title = f"{sample_name} - False Belief (Unchanged)"
            save_path = os.path.join(plots_dir, "cogmap_fb_unchanged.png")
            fb_unchanged_plot = create_cogmap_metrics_plot(fb_unchanged_per_turn, title, save_path=save_path, base_dir=self.out_dir)

        if isinstance(fog_probe_f1_per_turn, list):
            save_path = os.path.join(plots_dir, "fog_probe_f1.png")
            fog_probe_plots['f1'] = create_scalar_metric_plot(fog_probe_f1_per_turn, title=f"Fog Probe F1 per Turn - {sample_name}", y_label="F1", ylim=(0.0, 1.0), save_path=save_path, base_dir=self.out_dir)
        if isinstance(fog_probe_p_per_turn, list):
            save_path = os.path.join(plots_dir, "fog_probe_p.png")
            fog_probe_plots['p'] = create_scalar_metric_plot(fog_probe_p_per_turn, title=f"Fog Probe Precision per Turn - {sample_name}", y_label="Precision", ylim=(0.0, 1.0), save_path=save_path, base_dir=self.out_dir)
        if isinstance(fog_probe_r_per_turn, list):
            save_path = os.path.join(plots_dir, "fog_probe_r.png")
            fog_probe_plots['r'] = create_scalar_metric_plot(fog_probe_r_per_turn, title=f"Fog Probe Recall per Turn - {sample_name}", y_label="Recall", ylim=(0.0, 1.0), save_path=save_path, base_dir=self.out_dir)

        consistency_plots = {}
        if isinstance(pos_up_per_turn, list):
            save_path = os.path.join(plots_dir, "position_update.png")
            consistency_plots['pos_up'] = create_scalar_metric_plot(pos_up_per_turn, title=f"Position Update - {sample_name}", y_label="Score", ylim=(0.0, 1.0), save_path=save_path, base_dir=self.out_dir)
        if isinstance(fac_up_per_turn, list):
            save_path = os.path.join(plots_dir, "facing_update.png")
            consistency_plots['fac_up'] = create_scalar_metric_plot(fac_up_per_turn, title=f"Facing Update - {sample_name}", y_label="Score", ylim=(0.0, 1.0), save_path=save_path, base_dir=self.out_dir)
        if isinstance(pos_stab_per_turn, list):
            save_path = os.path.join(plots_dir, "position_stability.png")
            consistency_plots['pos_stab'] = create_scalar_metric_plot(pos_stab_per_turn, title=f"Position Stability - {sample_name}", y_label="Score", ylim=(0.0, 1.0), save_path=save_path, base_dir=self.out_dir)
        if isinstance(fac_stab_per_turn, list):
            save_path = os.path.join(plots_dir, "facing_stability.png")
            consistency_plots['fac_stab'] = create_scalar_metric_plot(fac_stab_per_turn, title=f"Facing Stability - {sample_name}", y_label="Score", ylim=(0.0, 1.0), save_path=save_path, base_dir=self.out_dir)

        # Display all plots in horizontal layout (up to 4 plots for samples)
        available_plots = []
        if infogain_plot:
            available_plots.append(("Information Gain per Turn", infogain_plot, "Information Gain per Turn"))
        if update_plot:
            available_plots.append(("Cognitive Map (Update)", update_plot, "Global Update Metrics"))
        if full_plot:
            available_plots.append(("Cognitive Map (Full)", full_plot, "Global Full Metrics"))
        if self_tracking_plot:
            available_plots.append(("Cognitive Map (Self-Tracking)", self_tracking_plot, "Global Self-Tracking Metrics"))
        if fb_unchanged_plot:
            available_plots.append(("FB CogMap (Unchanged)", fb_unchanged_plot, "False Belief CogMap Unchanged per Turn"))
        
        # Fog Probe
        if fog_probe_plots.get('f1'):
            available_plots.append(("Fog Probe F1", fog_probe_plots['f1'], "Fog Probe F1 per Turn"))
        if fog_probe_plots.get('p'):
            available_plots.append(("Fog Probe Precision", fog_probe_plots['p'], "Fog Probe Precision per Turn"))
        if fog_probe_plots.get('r'):
            available_plots.append(("Fog Probe Recall", fog_probe_plots['r'], "Fog Probe Recall per Turn"))
        
        # Cogmap FB plots (removed 'full' metric)
        if consistency_plots.get('pos_up'):
            available_plots.append(("Position Update", consistency_plots['pos_up'], "Position Update per Turn"))
        if consistency_plots.get('fac_up'):
            available_plots.append(("Facing Update", consistency_plots['fac_up'], "Facing Update per Turn"))
        if consistency_plots.get('pos_stab'):
            available_plots.append(("Position Stability", consistency_plots['pos_stab'], "Position Stability per Turn"))
        if consistency_plots.get('fac_stab'):
            available_plots.append(("Facing Stability", consistency_plots['fac_stab'], "Facing Stability per Turn"))


        if available_plots:
            f.write("<div class='cognitive-map-charts'>\n")
            f.write("<h3>📊 Performance Charts</h3>\n")
            f.write("<div class='plots-row'>\n")
            f.write("<div class='three-plots-grid'>\n")

            for title, plot_uri, alt_text in available_plots:
                plot_uri = self._relpath_from_html(plot_uri)
                f.write("<div class='plot-item'>\n")
                f.write(f"<h6>{title}</h6>\n")
                f.write(f"<img src='{plot_uri}' alt='{alt_text}' class='plot-image'>\n")
                f.write("</div>\n")

            f.write("</div>\n")  # End three-plots-grid
            f.write("</div>\n")  # End plots-row
            f.write("</div>\n")  # End cognitive-map-charts

    @staticmethod
    def _filter_eval_for_display(eval_dict: Dict) -> Dict:
        """Hide rot_dual from evaluation display (but keep it in metrics files)."""
        if not isinstance(eval_dict, dict) or not eval_dict:
            return eval_dict
        # Support both short-name and class-name keys, and a common typo.
        rot_keys = {"rot_dual", "rotdual", "RotDualEvaluationTask"}
        d = dict(eval_dict)
        per_task = d.get("per_task")
        if isinstance(per_task, dict):
            d["per_task"] = {k: v for k, v in per_task.items() if k not in rot_keys}
        task_metrics = d.get("task_metrics")
        if isinstance(task_metrics, dict):
            d["task_metrics"] = {k: v for k, v in task_metrics.items() if k not in rot_keys}
        return d

    def generate_sample_page(self, f, page_idx: int, sample_id: str, sample_data: Dict) -> None:
        """Generate a single sample page with combination selector"""
        f.write(f"<section class='sample-page' id='page{page_idx}'>\n")
        f.write(f"<h2>{escape(sample_id)}</h2>\n")

        # Add combination selector
        available_combos = [combo for combo in self.combinations
                           if combo in sample_data and sample_data[combo] is not None]

        # Always show combination selector for consistency, regardless of count
        if len(available_combos) >= 1:
            f.write("<div class='combination-selector'>\n")
            f.write("<h3>Select Configuration:</h3>\n")
            f.write("<div class='combo-buttons'>\n")
            for i, combo in enumerate(available_combos):
                active_class = "active" if i == 0 else ""
                f.write(f"<button class='combo-btn {active_class}' data-combo='{combo}' data-sample='{sample_id}' onclick=\"switchCombination('{combo}', '{sample_id}')\">{combo.replace('_', ' ').title()}</button>\n")
            f.write("</div>\n")
            f.write("</div>\n")

        # Single container for all combinations with seamless background
        f.write(f"<div class='combo-container' id='{sample_id}-container'>\n")

        # Store all combo data as JSON for dynamic switching
        f.write(f"<script type='application/json' id='{sample_id}-data'>\n")
        combo_data = {}
        for combo in available_combos:
            if combo in sample_data and sample_data[combo] is not None:
                combo_data[combo] = {
                    'html': self._generate_combo_html(sample_data[combo], combo, sample_id, page_idx)
                }
        f.write(json.dumps(combo_data, ensure_ascii=False))
        f.write("</script>\n")

        # Generate initial content (first combo)
        if available_combos:
            initial_combo = available_combos[0]
            entry = sample_data[initial_combo]

            # Generate content that will be replaced dynamically
            f.write(f"<div class='combo-content-inner' id='{sample_id}-content'>\n")
            f.write(self._generate_combo_html(entry, initial_combo, sample_id, page_idx))
            f.write("</div>\n")

        f.write("</div>\n")  # End combo-container
        f.write("</section>\n")

    def _generate_combo_html(self, entry: Dict, combo: str, sample_id: str, page_idx: int) -> str:
        """Generate HTML content for a single combo configuration"""
        from io import StringIO
        output = StringIO()

        # Generate Sample Metrics at the top
        self.generate_sample_metrics(output, entry, f"{combo} {sample_id}")

        # Generate Performance Charts (Information Gain + Cognitive Map plots)
        self.generate_cognitive_map_charts(output, entry, f"{combo} {sample_id}", sample_id)

        # Display initial room image if available
        if self.show_images and entry.get("initial_room_image"):
            img_name = self._relpath_from_html(entry["initial_room_image"])
            output.write(f"<img src='{img_name}' class='room' alt='Initial room state'>\n")

        # For passive runs, show prompt+images ONCE (do not repeat per question).
        passive_ctx = self._load_passive_prompt_context(entry)
        if passive_ctx:
            output.write("<div class='section-header'><h3>🧾 Passive Exploration Context</h3></div>\n")
            sys_p = passive_ctx.get("system", "") if isinstance(passive_ctx, dict) else ""
            user_p = passive_ctx.get("user", "") if isinstance(passive_ctx, dict) else ""
            imgs = passive_ctx.get("images", []) if isinstance(passive_ctx, dict) else []
            if sys_p:
                self._render_expandable_block(output, sys_p, f"passive_sys_{page_idx}_{combo}", "🧩 System Prompt")
            if user_p:
                self._render_expandable_block(output, user_p, f"passive_user_{page_idx}_{combo}", "📝 Prompt (before evaluation question)")
            if self.show_images and imgs:
                output.write("<div class='question-right'>\n")
                for i, p in enumerate(imgs):
                    if isinstance(p, str):
                        output.write(
                            f"<figure><img src='{p}' class='room-plot' alt='Passive context image {i + 1}'>"
                            f"<figcaption>Context Image {i + 1}</figcaption></figure>\n"
                        )
                output.write("</div>\n")

        # Environment config
        # Generate exploration turns and evaluation tasks
        self.generate_exploration_turns(output, entry, page_idx)

        return output.getvalue()

    def _render_expandable_block(self, f, content: str, block_id: str, title: str, block_class: str = "user") -> None:
        """Helper to render expandable content blocks"""
        if not content:
            return
        content_short = escape(content[:300]).replace("\n", "<br>")
        content_full = escape(content).replace("\n", "<br>")
        f.write(f"<div id='{block_id}' class='block {block_class} expandable' onclick='toggleObservation(\"{block_id}\")' data-expanded='false'><strong>{title} <span class='expand-hint'>(click to toggle)</span></strong><br><span class='content-text'>{content_short}...</span></div>\n")
        f.write(f"<div id='{block_id}_full' style='display:none'>{content_full}</div>\n")
        f.write(f"<div id='{block_id}_short' style='display:none'>{content_short}...</div>\n")

    def _render_simple_block(self, f, content: str, title: str, block_class: str) -> None:
        """Helper to render simple content blocks"""
        if not content:
            return
        content_escaped = escape(content).replace("\n", "<br>")
        f.write(f"<div class='block {block_class}'><strong>{title}</strong><br>{content_escaped}</div>\n")

    def _render_cogmap_responses(
        self,
        f,
        cogmap_log: Dict,
        page_idx: int,
        t_idx: int,
        env_log: Optional[Dict] = None,
        show_gt_observed: bool = True,
    ) -> None:
        """Helper to render cognitive map responses"""
        def _safe_id(s: object) -> str:
            return ''.join(c if str(c).isalnum() else '_' for c in str(s))

        cogmap_types = [
            ('global', '🗺️ Global Cognitive Map Response'),
            ('local', '🗺️ Local Cognitive Map Response'),
            ('local_newly', '🗺️ Local (Newly Observed) Map Response'),
            ('fog_probe', '🌫️ Fog Probe Response'),
        ]

        kind = "turn"
        turn_num = t_idx
        qid = ""
        if isinstance(env_log, dict):
            if env_log.get("false_belief_log"):
                kind = "fb"
            elif env_log.get("evaluation_log"):
                kind = "eval"
                qid = str((((env_log.get("evaluation_log") or {}).get("evaluation_data") or {}).get("id")) or "")
            elif env_log.get("is_exploration_phase", False):
                kind = "exp"
            turn_num = env_log.get("turn_number", t_idx)

        for map_type, title in cogmap_types:
            data = cogmap_log.get(map_type, {})
            if data.get('original_response'):
                # Skip duplicate response display for newly observed (same as local)
                if map_type != 'local_newly':
                    tail = f"{kind}_{page_idx}_{turn_num}"
                    if qid:
                        tail = f"{tail}_{_safe_id(qid)}"
                    response_id = f"cogmap_{map_type}_{tail}"
                    self._render_expandable_block(f, data['original_response'], response_id, title, "cogmap-response")

                # Add Symbolic Map and Fog Probe Image side-by-side
                if map_type == 'fog_probe':
                    f.write("<div class='fog-probe-container' style='display: flex; gap: 20px;'>\n")
                    
                    # Left: Symbolic Map
                    symbolic_map = data.get('symbolic_map')
                    if symbolic_map:
                        f.write(f"<div class='symbolic-map-box' style='flex: 1;'>")
                        f.write(f"<strong>🗺️ Symbolic Fog Map</strong>")
                        f.write(f"<pre class='symbolic-map' style='font-family: monospace; white-space: pre; overflow-x: auto;'>{escape(symbolic_map)}</pre>")
                        f.write(f"</div>\n")
                    
                    # Right: Fog Probe Image
                    if self.show_images:
                        msg_imgs = (data.get("message_images") or []) if isinstance(data, dict) else []
                        if not msg_imgs and env_log:
                            msg_imgs = env_log.get("message_images") or []
                        for img_path in msg_imgs:
                            if not isinstance(img_path, str):
                                continue
                            img_src = self._relpath_from_html(img_path)
                            if "top_down_candidates" in img_src:
                                f.write("<div class='fog-probe-image' style='flex: 1;'>")
                                f.write(f"<figure><img src='{img_src}' class='room-plot' alt='Fog Probe Candidates' style='max-width: 100%;'><figcaption>Fog Probe Candidates</figcaption></figure>")
                                f.write("</div>\n")
                                break  # Only show the first matching image
                    
                    f.write("</div>\n") # End fog-probe-container

                # Add JSON display for global and local
                if map_type == 'global':
                    # For global, display pred_json and GT (full). Optionally show GT (observed).
                    pred_json = data.get('pred_json', {})
                    gt_json = data.get('gt_json', {})
                    gt_json_full = data.get('gt_json_full', {})

                    if pred_json or gt_json or gt_json_full:
                        f.write("<div class='json-container global'>\n")
                        f.write("<div class='json-header'>")
                        f.write("<strong>📊 Cognitive Map JSONs</strong>")
                        f.write("</div>\n")
                        f.write("<div class='json-content'>\n")
                        f.write("<div class='json-compare global'>\n")

                        # Left - pred_json
                        f.write("<div class='json-box left predicted'>\n")
                        f.write("<strong>🤖 Predicted</strong>\n")
                        if pred_json:
                            f.write("<div class='json-content-inner'>\n")
                            f.write(f"<pre>{escape(json.dumps(pred_json, indent=2))}</pre>\n")
                            f.write("</div>\n")
                        else:
                            f.write("<div class='empty-json'>(no data)</div>\n")
                        f.write("</div>\n")

                        # Middle - gt_json (optional)
                        if show_gt_observed:
                            f.write("<div class='json-box middle gt-observed'>\n")
                            f.write("<strong>🎯 Ground Truth (Observed)</strong>\n")
                            if gt_json:
                                f.write("<div class='json-content-inner'>\n")
                                f.write(f"<pre>{escape(json.dumps(gt_json, indent=2))}</pre>\n")
                                f.write("</div>\n")
                            else:
                                f.write("<div class='empty-json'>(no data)</div>\n")
                            f.write("</div>\n")

                        # Right - gt_json_full
                        f.write("<div class='json-box right gt-full'>\n")
                        f.write("<strong>🎯 Ground Truth (Full)</strong>\n")
                        if gt_json_full:
                            f.write("<div class='json-content-inner'>\n")
                            f.write(f"<pre>{escape(json.dumps(gt_json_full, indent=2))}</pre>\n")
                            f.write("</div>\n")
                        elif gt_json:
                            # Fallback if older logs only have observed GT
                            f.write("<div class='json-content-inner'>\n")
                            f.write(f"<pre>{escape(json.dumps(gt_json, indent=2))}</pre>\n")
                            f.write("</div>\n")
                        else:
                            f.write("<div class='empty-json'>(no data)</div>\n")
                        f.write("</div>\n")

                        f.write("</div>\n")  # End json-compare
                        f.write("</div>\n")  # End json-content
                        f.write("</div>\n")  # End json-container

                elif map_type in ('local', 'local_newly'):
                    # For local, display pred_json and gt_json in two columns
                    pred_json = data.get('pred_json', {})
                    gt_json = data.get('gt_json', {})

                    if pred_json or gt_json:
                        # Use 'local' class for styling compatibility
                        style_class = 'local'
                        f.write(f"<div class='json-container {style_class}'>\n")
                        f.write("<div class='json-header'>")
                        f.write("<strong>📊 Cognitive Map JSONs</strong>")
                        f.write("</div>\n")
                        f.write("<div class='json-content'>\n")
                        f.write(f"<div class='json-compare {style_class}'>\n")

                        # Left - pred_json
                        f.write("<div class='json-box left predicted'>\n")
                        f.write("<strong>🤖 Predicted</strong>\n")
                        if pred_json:
                            f.write("<div class='json-content-inner'>\n")
                            f.write(f"<pre>{escape(json.dumps(pred_json, indent=2))}</pre>\n")
                            f.write("</div>\n")
                        else:
                            f.write("<div class='empty-json'>(no data)</div>\n")
                        f.write("</div>\n")

                        # Right - gt_json
                        f.write("<div class='json-box right gt'>\n")
                        f.write("<strong>🎯 Ground Truth</strong>\n")
                        if gt_json:
                            f.write("<div class='json-content-inner'>\n")
                            f.write(f"<pre>{escape(json.dumps(gt_json, indent=2))}</pre>\n")
                            f.write("</div>\n")
                        else:
                            f.write("<div class='empty-json'>(no data)</div>\n")
                        f.write("</div>\n")

                        f.write("</div>\n")  # End json-compare
                        f.write("</div>\n")  # End json-content
                        f.write("</div>\n")  # End json-container

                elif map_type == 'fog_probe':
                    all_candidate_points = data.get('all_candidate_points', [])
                    pred_points = data.get('pred_points', [])
                    correct_points = data.get('correct_points', [])

                    if all_candidate_points:
                        pt_to_label = {str(pt): chr(ord('A') + i) for i, pt in enumerate(all_candidate_points)}
                        candidates_labels = [chr(ord('A') + i) for i in range(len(all_candidate_points))]
                        
                        pred_labels_list = []
                        if pred_points:
                            for pt in pred_points:
                                lbl = pt_to_label.get(str(pt))
                                if lbl: pred_labels_list.append(lbl)
                        
                        correct_labels_list = []
                        if correct_points:
                            for pt in correct_points:
                                lbl = pt_to_label.get(str(pt))
                                if lbl: correct_labels_list.append(lbl)

                        f.write(f"<div class='json-container {map_type}'>\n")
                        f.write("<div class='json-header'>")
                        f.write(f"<strong>🔍 {map_type.replace('_', ' ').title()} JSONs</strong>")
                        f.write("</div>\n")
                        
                        # Compact display
                        f.write("<div class='json-content' style='padding: 5px 10px;'>\n")
                        f.write(f"<div style='margin-bottom:2px;'><strong>📍 Candidates:</strong> {', '.join(candidates_labels)}</div>\n")
                        f.write(f"<div style='margin-bottom:2px;'><strong>🤖 Predicted:</strong> {', '.join(pred_labels_list) if pred_labels_list else '(none)'}</div>\n")
                        f.write(f"<div><strong>🎯 Ground Truth:</strong> {', '.join(correct_labels_list) if correct_labels_list else '(none)'}</div>\n")
                        f.write("</div>\n")
                        f.write("</div>\n")

    def _render_cogmap_metrics(self, f, cogmap_log: Dict) -> None:
        """Helper to render cognitive map metrics"""
        if not cogmap_log:
            return

        # Extract metrics
        global_log = cogmap_log.get("global", {})
        local_log = cogmap_log.get("local", {})
        local_newly_log = cogmap_log.get("local_newly", {})

        metrics_block = {
            "Global": global_log.get("metrics", {}) if global_log else {},
            "Global (Full)": global_log.get("metrics_full", {}) if global_log else {},
            "Local": local_log.get("metrics", {}) if local_log else {},
            "Local (Newly)": local_newly_log.get("metrics", {}) if local_newly_log else {},
            "Fog Probe": cogmap_log.get("fog_probe", {}).get("metrics", {}) if cogmap_log.get("fog_probe") else {},
        }

        if any(metrics_block.values()):
            f.write("<div class='block cogmap'><strong>🧠 Cognitive Map Metrics</strong>")
            f.write(VisualizationHelper.dict_to_html(metrics_block))
            f.write("</div>\n")

    def _render_turn_metrics(self, f, env_log: Dict) -> None:
        """Helper to render turn metrics"""
        metrics = {}

        if env_log['is_exploration_phase'] and env_log.get('exploration_log'):
            exp_log = env_log['exploration_log']
            metrics.update({
                "node_coverage": exp_log.get('node_coverage'),
                "edge_coverage": exp_log.get('edge_coverage'),
                "is_action_fail": exp_log.get('is_action_fail'),
                "step": exp_log.get('step'),
                "action_counts": exp_log.get('action_counts'),
                "information_gain": exp_log.get('information_gain')
            })

        if env_log.get('info'):
            metrics.update(env_log['info'])

        if metrics:
            f.write("<div class='metrics'><strong>📈 Turn Metrics</strong>")
            f.write(VisualizationHelper.dict_to_html(metrics))
            f.write("</div>\n")

    def _render_turn_images(self, f, env_log: Dict, env_turn_logs: List, t_idx: int) -> None:
        """Helper to render turn images"""
        f.write("<div class='turn-right'>\n")

        if self.show_images:
            # Previous image (initial if first turn)
            if t_idx > 0:
                prev_img = env_turn_logs[t_idx-1].get('room_image')
                if prev_img:
                    prev_img = self._relpath_from_html(prev_img)
                    f.write(f"<figure><img src='{prev_img}' class='room-plot' alt='Previous state'><figcaption>State before Turn {t_idx+1}</figcaption></figure>\n")

            # Current image
            curr_img = env_log.get('room_image')
            if curr_img:
                curr_img = self._relpath_from_html(curr_img)
                f.write(f"<figure><img src='{curr_img}' class='room-plot' alt='Current state'><figcaption>State at Turn {t_idx+1}</figcaption></figure>\n")

            # Message images
            if 'message_images' in env_log:
                for img_idx, img_path in enumerate(env_log['message_images']):
                    if isinstance(img_path, str):
                        img_path = self._relpath_from_html(img_path)
                        f.write(f"<figure><img src='{img_path}' class='room-plot' alt='Environment image {img_idx + 1}'><figcaption>Observation {img_idx + 1}</figcaption></figure>\n")

        f.write("</div>\n")  # End turn-right


    def generate_exploration_turns(self, f, entry: Dict, page_idx: int) -> None:
        """Generate exploration turn logs and evaluation tasks"""
        env_turn_logs = entry.get("env_turn_logs", [])
        evaluation_tasks = entry.get("evaluation_tasks", {}) or {}
        false_belief_turn_logs = entry.get("false_belief_turn_logs", [])
        is_passive = self._is_passive_combo(entry)

        extra_eval_modes = {}
        for k, v in entry.items():
            if k.startswith("evaluation_tasks_"):
                mode = k.split("evaluation_tasks_", 1)[1] or "default"
                extra_eval_modes[mode] = v
        eval_groups = [("default", evaluation_tasks)] + sorted(extra_eval_modes.items())

        if not env_turn_logs and not evaluation_tasks and not false_belief_turn_logs:
            f.write("<div class='metrics'><strong>⚠️ No turns available</strong></div>\n")
            return

        # Helper to render a list of logs
        def render_logs(logs, title_prefix="Turn"):
            for t_idx, env_log in enumerate(logs):
                turn_num = env_log.get('turn_number', t_idx)
                safe_prefix = ''.join(c if str(c).isalnum() else '_' for c in str(title_prefix))
                f.write("<div class='turn-split'>\n")
                f.write(f"<h3>🔄 {title_prefix} {turn_num}</h3>\n")
                f.write("<div class='turn-content'>\n")

                # Left side: conversation and metrics
                f.write("<div class='turn-left'>\n")

                # Display user message (environment observation)
                if env_log.get('user_message'):
                    # Use unique ID based on log content hash or index to avoid collisions
                    obs_id = f"obs_{page_idx}_{safe_prefix}_{t_idx}"
                    self._render_expandable_block(f, env_log['user_message'], obs_id, "👤 Environment Observation")

                # Display assistant thinking and action
                if env_log.get('assistant_think_message'):
                    think_id = f"think_{page_idx}_{safe_prefix}_{t_idx}"
                    self._render_expandable_block(f, env_log['assistant_think_message'], think_id, "🤔 Assistant Thinking", "think")
                self._render_simple_block(f, env_log.get('assistant_parsed_message', ''), "💬 Assistant Action", "answer")

                # Determine cogmap_log source (EnvTurnLog or FBLog)
                cogmap_log = env_log.get('cogmap_log')
                if not cogmap_log and env_log.get('false_belief_log'):
                     raw_fb_log = env_log['false_belief_log'].get('cogmap_log')
                     if raw_fb_log:
                         # Wrap FB log to be compatible with _render_cogmap_responses
                         cogmap_log = dict(raw_fb_log)
                         if 'original_response' in cogmap_log and 'global' not in cogmap_log:
                             cogmap_log['global'] = {'original_response': cogmap_log['original_response']}

                # Display cognitive map original responses if available
                if cogmap_log:
                    self._render_cogmap_responses(f, cogmap_log, page_idx, t_idx, env_log=env_log)
                    # Render standard metrics only if not FB (FB has custom block)
                    if "changed_objects_per_object" not in cogmap_log:
                        self._render_cogmap_metrics(f, cogmap_log)

                # Display turn metrics
                self._render_turn_metrics(f, env_log)

                # Show cogmap metric for false belief turns
                if cogmap_log and "changed_objects_per_object" in cogmap_log:
                    cm_log = cogmap_log
                    
                    # Note: changed_objects is now per-object, so we don't render a single response
                    # Instead, we'll show the per-object metrics
                    
                    f.write("<div class='metrics'><strong>🧭 False Belief Cogmap Metrics</strong>")
                    metrics_block = {}
                    
                    # Display per-object metrics for changed objects
                    per_obj_metrics = cm_log.get('changed_objects_per_object', {})
                    ret_per_obj = cm_log.get('retention_per_object', {})
                    if per_obj_metrics:
                        for obj_name, obj_metrics in per_obj_metrics.items():
                            metrics_block[f'Changed: {obj_name}'] = obj_metrics
                            ret_m = ret_per_obj.get(obj_name)
                            if ret_m:
                                metrics_block[f'Retention: {obj_name}'] = ret_m
                    
                    if cm_log.get('unchanged_objects'):
                         metrics_block['Unchanged (all)'] = cm_log['unchanged_objects'].get('global', {}).get('metrics')
                    f.write(VisualizationHelper.dict_to_html(metrics_block))
                    f.write("</div>\n")

                    # Compact per-turn object set view
                    # Use new key names: 'all_changed_object_names' and 'newly_observed_changed_objects'
                    all_changed_names = list((cm_log.get('all_changed_object_names') or [])) if isinstance(cm_log, dict) else []
                    newly_observed_changed = list((cm_log.get('newly_observed_changed_objects') or [])) if isinstance(cm_log, dict) else []
                    unchanged_names = list((cm_log.get('unchanged_object_names') or [])) if isinstance(cm_log, dict) else []
                    full_names = sorted(set([str(x) for x in all_changed_names + unchanged_names if x is not None]))

                    # Get predicted object names from per-object metrics
                    per_obj_metrics = cm_log.get('changed_objects_per_object', {})
                    pred_changed_keys = sorted([str(x) for x in newly_observed_changed if x is not None])
                    
                    def _pred_unchanged_obj_names() -> List[str]:
                        sub = cm_log.get('unchanged_objects') or {}
                        pred = ((sub.get("global") or {}).get("pred_json") or {}) if isinstance(sub, dict) else {}
                        if not isinstance(pred, dict):
                            return []
                        return sorted([str(n) for n in pred.keys() if n != "agent"])

                    obj_block = {
                        "all_objects": full_names,
                        "all_changed_objects": sorted([str(x) for x in all_changed_names if x is not None]),
                        "newly_observed_changed (this turn)": sorted([str(x) for x in newly_observed_changed if x is not None]),
                        "unchanged_objects": sorted([str(x) for x in unchanged_names if x is not None]),
                        "pred_changed_keys": pred_changed_keys,
                        "pred_unchanged_keys": _pred_unchanged_obj_names(),
                    }
                    f.write("<div class='metrics'><strong>🧾 False Belief Object Sets</strong>")
                    f.write(VisualizationHelper.dict_to_html(obj_block))
                    f.write("</div>\n")

                if env_log.get('false_belief_log'):
                    fb_log = env_log['false_belief_log'] or {}
                    fb_info = dict(fb_log) if isinstance(fb_log, dict) else {}
                    fb_info.pop('room_state', None)
                    fb_info.pop('agent_state', None)
                    fb_info.pop('cogmap_log', None)
                    f.write("<div class='metrics'><strong>🧭 False Belief Info</strong>")
                    f.write(VisualizationHelper.dict_to_html(fb_info))
                    f.write("</div>\n")

                f.write("</div>\n")  # End turn-left

                # Right side: room and message images
                self._render_turn_images(f, env_log, logs, t_idx)
                f.write("</div>\n")  # End turn-content
                f.write("</div>\n")  # End turn-split

        # Render exploration turns
        if env_turn_logs:
            f.write("<div class='section-header'><h3>🌍 Exploration Phase</h3></div>\n")
            render_logs(env_turn_logs)

        if false_belief_turn_logs:
            f.write("<div class='section-header'><h3>🧭 False Belief Exploration</h3></div>\n")
            render_logs(false_belief_turn_logs, "FB Turn")

        # Generate evaluation turns if available
        eval_offset = len(env_turn_logs)
        for mode_label, task_dict in eval_groups:
            if not task_dict:
                continue
            mode_title = "default" if mode_label == "default" else mode_label
            f.write(f"<div class='section-header'><h3>📊 Evaluation ({escape(mode_title)})</h3></div>\n")
            # Handle new nested structure: {task_type: {question_id: eval_data}}
            for eval_idx, (task_type, task_questions) in enumerate(task_dict.items()):
                t_idx = eval_offset + eval_idx

                # Create evaluation turn section with task selector data
                task_label = self._format_eval_task_label(mode_label, task_type)
                task_key = f"{mode_label}:{task_type}"
                f.write(
                    f"<div class='turn-split eval-task' data-task-name='{escape(task_label)}' "
                    f"data-task-key='{escape(task_key)}' data-task-type='{escape(task_type)}' "
                    f"data-eval-mode='{escape(mode_label)}'>\n"
                )
                f.write(f"<h3>📊 Task: {escape(task_type)}</h3>\n")

                f.write("<div class='turn-content'>\n")

                # Left side: conversation and metrics
                f.write("<div class='turn-left'>\n")

                for question_idx, (question_id, eval_log) in enumerate(task_questions.items()):
                    f.write(f"<div class='question-section' data-question-id='{question_id}'>\n")
                    f.write(f"<h4>Question {question_idx + 1} (ID: {question_id})</h4>\n")

                    # Create a split layout for this question
                    f.write("<div class='question-content'>\n")
                    f.write("<div class='question-left'>\n")

                    # Display evaluation question (prefer stored user_message; fallback to evaluation_data.question)
                    q_text = eval_log.get("user_message") or ""
                    if not q_text:
                        q_text = ((eval_log.get("evaluation_log") or {}).get("evaluation_data") or {}).get("question") or ""
                    if is_passive and q_text and "## Evaluation Question" in q_text:
                        q_text = "## Evaluation Question\n" + q_text.split("## Evaluation Question", 1)[1].strip()
                    if q_text:
                        obs_id = f"obs_{page_idx}_{t_idx}_{question_idx}"
                        self._render_expandable_block(f, q_text, obs_id, "❓ Evaluation Question")
                    # Display assistant thinking and action
                    if eval_log.get('assistant_raw_message'):
                        think_id = f"think_{page_idx}_{t_idx}_{question_idx}"
                        self._render_expandable_block(f, eval_log['assistant_raw_message'], think_id, "🤔 Assistant Answer", "answer")
                    
                    # self._render_simple_block(f, eval_log.get('assistant_parsed_message', ''), "💬 Assistant Answer", "answer")
                    if eval_log.get('cogmap_log'):
                        self._render_cogmap_responses(f, eval_log['cogmap_log'], page_idx, t_idx, env_log=eval_log)
                        self._render_cogmap_metrics(f, eval_log['cogmap_log'])
                    # Display evaluation results
                    if eval_log.get('evaluation_log'):
                        eval_info = eval_log['evaluation_log']
                        f.write("<div class='block evaluation'><strong>✅ Evaluation Results</strong>")
                        details = {
                            **eval_info.get("evaluation_data", {}),
                            **eval_info.get("evaluation_info", {}),
                            "score": float(eval_info.get("score")),
                            "evaluation_mode": mode_label,
                        }
                        f.write(VisualizationHelper.dict_to_html(details))
                        f.write("</div>\n")

                    f.write("</div>\n")  # End question-left

                    # Right side: images for this specific question
                    f.write("<div class='question-right'>\n")
                    if self.show_images:
                        # Current evaluation state image
                        if eval_log.get("room_image"):
                            img_name = self._relpath_from_html(eval_log["room_image"])
                            f.write(f"<figure><img src='{img_name}' class='room-plot' alt='Evaluation state'><figcaption>Q{question_idx + 1}: {escape(task_type)}</figcaption></figure>\n")

                        # For passive exploration runs: do NOT repeat exploration-history images per question.
                        # Only show the question-specific image for vision tasks (builder appends it last).
                        msg_imgs = eval_log.get("message_images") or []
                        if is_passive:
                            # Try to find the image for the question (usually the last one for vision tasks)
                            if msg_imgs:
                                img_path = msg_imgs[-1]
                                if isinstance(img_path, str):
                                    img_path = self._relpath_from_html(img_path)
                                    f.write(f"<figure><img src='{img_path}' class='room-plot' alt='Question image'><figcaption>Question Image</figcaption></figure>\n")
                        else:
                            for img_idx, img_path in enumerate(msg_imgs):
                                if isinstance(img_path, str):
                                    img_path = self._relpath_from_html(img_path)
                                    f.write(f"<figure><img src='{img_path}' class='room-plot' alt='Evaluation image {img_idx + 1}'><figcaption>Q{question_idx + 1} Image {img_idx + 1}</figcaption></figure>\n")

                    f.write("</div>\n")  # End question-right
                    f.write("</div>\n")  # End question-content
                    f.write("</div>\n")  # End question-section

                f.write("</div>\n")  # End turn-left
                f.write("</div>\n")  # End turn-content
                f.write("</div>\n")  # End turn-split
            eval_offset += len(task_dict)

        f.write("</section>\n")

    def _write_html_file(self, output_html: str, page_idx: int, page_urls: List[str], render_fn) -> None:
        self.html_dir = os.path.dirname(output_html)
        os.makedirs(self.html_dir, exist_ok=True)
        self.page_urls = page_urls
        self.current_page = page_idx

        js_code = (
            JAVASCRIPT_CODE
            .replace('{total_pages}', str(self.total_pages))
            .replace('{page_urls}', json.dumps(page_urls))
            .replace('{current_page}', str(page_idx))
        )
        with open(output_html, "w") as f:
            f.write(HTML_TEMPLATE.format(
                model_name=escape(self.meta.get('model_name', 'Unknown Model')),
                total_pages=self.total_pages,
                css_styles=CSS_STYLES,
                javascript_code=js_code
            ))
            render_fn(f)
            f.write("</body></html>")

    def generate_html(self) -> str:
        """Generate the HTML files with summary + per-sample pages"""
        page_paths = [self.output_html]
        for sample_id, sample_data in self.flat:
            sample_subdir = self._get_sample_subdir(sample_id, sample_data)
            sample_html = os.path.join(self.out_dir, sample_subdir, "html", "index.html")
            page_paths.append(sample_html)

        for page_idx, output_html in enumerate(page_paths):
            page_urls = [os.path.relpath(p, os.path.dirname(output_html)) for p in page_paths]
            if page_idx == 0:
                self._write_html_file(output_html, page_idx, page_urls, self.generate_toc_page)
            else:
                sample_id, sample_data = self.flat[page_idx - 1]
                self._write_html_file(
                    output_html,
                    page_idx,
                    page_urls,
                    lambda f, s_id=sample_id, s_data=sample_data, idx=page_idx: self.generate_sample_page(f, idx, s_id, s_data),
                )

        return self.output_html

