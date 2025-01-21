# Visualization Style Settings

This guide explains the available style settings for visualizations in the analysis tools.

## Color Palettes

The analysis tools provide several predefined color palettes for consistent visualization:

- `style.colors_names`
- `style.colors_light`
- `style.colors_dark`

## Plot Settings

The default plot settings use a dark theme:

```python
template = "plotly_dark"      # Plotly dark theme
plot_bgcolor = "#000000"      # Black plot background
paper_bgcolor = "#000000"     # Black paper background
```

## Usage in Visualizations

The style settings are automatically applied in visualization functions like `plot_training_curves` and `plot_learning_rates`. The color palettes are used cyclically when plotting multiple runs:

- Training curves use `colors_light`
- Validation curves use `colors_dark`
- Special visualizations can use specific colors from `colors_names`
