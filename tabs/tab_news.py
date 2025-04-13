# tabs/tab_news.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
# FIX: Import Any
from typing import Optional, Dict, Any

def render_tab_news(st_tab: Any, news_api_enabled: bool): # FIX: Keep type hint
    """Renders the content for the News tab."""
    with st_tab:
        st.header("Recent News & Sentiment Analysis")
        if news_api_enabled:
            keywords_used = st.session_state.get('news_keywords_processed', 'N/A')
            st.markdown(f"Displaying news for keywords: **'{keywords_used}'**")
            st.caption(f"Status: *{st.session_state.get('news_status', 'Not fetched')}*")

            df_news_display = st.session_state.get('news_df', pd.DataFrame())

            if not df_news_display.empty:
                def style_sentiment(v):
                    try: val = float(v)
                    except (ValueError, TypeError): return ''
                    if val <= -0.05: return 'background-color: #FFDDDD; color: #A00000;'
                    elif val >= 0.05: return 'background-color: #DDFFDD; color: #006400;'
                    else: return ''

                df_display_prep = df_news_display[['Pub', 'Src', 'Title', 'Sent', 'URL']].copy()
                df_display_prep.rename(columns={'Pub': 'Published', 'Src': 'Source', 'Sent': 'Sentiment'}, inplace=True)
                df_display_prep['Sentiment_Style'] = df_display_prep['Sentiment']
                df_display_prep['Sentiment'] = df_display_prep['Sentiment'].map('{:.3f}'.format)

                st.dataframe(
                    df_display_prep.style.apply(lambda x: x.map(style_sentiment), subset=['Sentiment_Style']),
                    column_config={"URL": st.column_config.LinkColumn("Link", display_text="🔗", width="small"),
                                   "Title": st.column_config.TextColumn("Title", width="large"),
                                   "Published": st.column_config.TextColumn("Published", width="medium"),
                                   "Source": st.column_config.TextColumn("Source", width="medium"),
                                   "Sentiment": st.column_config.NumberColumn("Sentiment", width="small", help="VADER Score [-1 to +1]", format="%.3f"),
                                   "Sentiment_Style": None},
                    hide_index=True, use_container_width=True, height=450
                )

                st.subheader("Sentiment Score Distribution")
                if 'Sent' in df_news_display.columns and pd.api.types.is_numeric_dtype(df_news_display['Sent']):
                    fig_hist = go.Figure(data=[go.Histogram(x=df_news_display['Sent'], nbinsx=10, marker_color='#1f77b4', opacity=0.75)])
                    fig_hist.update_layout(title_text="Distribution of VADER Scores", xaxis_title="Score", yaxis_title="Count", bargap=0.1, height=350, margin=dict(t=40, b=40, l=40, r=20))
                    st.plotly_chart(fig_hist, use_container_width=True)
                else: st.warning("Sentiment data missing/invalid.")
            else: st.info("No news articles found or processed. Fetch news via sidebar.")
        else: st.error("News Feed Disabled (check API key in secrets).")
