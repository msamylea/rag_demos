import gradio as gr

theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="teal",
    neutral_hue="stone",
    radius_size="lg",
    font=[gr.themes.GoogleFont('Poppins'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
).set(
    body_background_fill='*primary_200',
    background_fill_primary='*primary_400',
    background_fill_secondary='*border_color_accent_subdued',
    border_color_accent='*neutral_300',
    border_color_primary='*primary_600',
    color_accent='*primary_300',
    link_text_color='*secondary_500',
    link_text_color_dark='*secondary_100',
    shadow_drop='*shadow_drop_lg',
    shadow_drop_lg='*shadow_drop',
    shadow_inset='*shadow_drop_lg',
    shadow_spread='12px',
    shadow_spread_dark='10px',
    block_background_fill='*primary_50',
    block_border_width='1px',
    block_info_text_color='*body_text_color',
    block_info_text_size='*text_md'
)
