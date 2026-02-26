"""
🎮 클랜 종합 분석기 (Clan Analyzer)
Clash of Clans 클랜의 생존 확률과 리그 등급을 예측하는 통합 웹 앱

실행 방법: streamlit run app_unified.py
"""
import streamlit as st
import joblib
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="클랜 종합 분석기",
    page_icon="⚔️",
    layout="centered"
)

# ==========================================
# 모델 로드
# ==========================================
@st.cache_resource
def load_survival_models():
    """클랜 생존 예측 모델 로드"""
    model = joblib.load('clan_retention_model.pkl')
    war_freq_encoder = joblib.load('war_frequency_encoder.pkl')
    clan_type_encoder = joblib.load('clan_type_encoder.pkl')
    return model, war_freq_encoder, clan_type_encoder

@st.cache_resource
def load_league_models():
    """리그 등급 예측 모델 로드"""
    model = joblib.load('league_prediction_model.pkl')
    label_encoder = joblib.load('league_label_encoder.pkl')
    tier_standards = joblib.load('tier_standards.pkl')
    return model, label_encoder, tier_standards

# 모델 로드
survival_model, war_freq_encoder, clan_type_encoder = load_survival_models()
league_model, league_encoder, tier_standards = load_league_models()

# ==========================================
# 메인 헤더
# ==========================================
st.title("⚔️ 클랜 종합 분석기")
st.markdown("**Clash of Clans 클랜의 생존 확률과 리그 등급을 예측합니다**")
st.markdown("---")

# ==========================================
# 탭 구성
# ==========================================
tab1, tab2 = st.tabs(["🛡️ 클랜 생존 예측", "🏆 리그 등급 예측"])

# ==========================================
# 탭 1: 클랜 생존 예측
# ==========================================
with tab1:
    st.subheader("🛡️ 클랜 생존 예측기")
    st.markdown("당신의 클랜은 앞으로도 살아남을 수 있을까요?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mean_member_trophies = st.number_input(
            "멤버 평균 트로피",
            min_value=0, max_value=6000, value=1500,
            help="클랜원들의 평균 트로피 점수",
            key="survival_trophies"
        )
        
        mean_member_level = st.number_input(
            "멤버 평균 레벨",
            min_value=1, max_value=300, value=100,
            help="클랜원들의 평균 경험치 레벨",
            key="survival_level"
        )
        
        required_trophies = st.number_input(
            "가입 조건 트로피",
            min_value=0, max_value=5500, value=800,
            help="클랜 가입에 필요한 최소 트로피",
            key="survival_required"
        )
    
    with col2:
        war_frequency = st.selectbox(
            "전쟁 빈도 설정",
            options=['always', 'moreThanOncePerWeek', 'oncePerWeek', 'lessThanOncePerWeek', 'never', 'unknown'],
            index=0,
            help="클랜의 전쟁 빈도 설정값",
            key="survival_war_freq"
        )
        
        clan_type = st.selectbox(
            "클랜 공개 설정",
            options=['inviteOnly', 'open', 'closed'],
            index=0,
            help="클랜의 가입 방식",
            key="survival_clan_type"
        )
        
        is_family_friendly = st.checkbox(
            "가족 친화 모드",
            value=True,
            help="가족 친화 설정 여부",
            key="survival_family"
        )
    
    if st.button("🔍 생존 확률 확인", type="primary", use_container_width=True, key="survival_btn"):
        # 파생변수 계산
        activity_ratio = mean_member_trophies / (mean_member_level + 1)
        entry_gap = mean_member_trophies - required_trophies
        
        # 인코딩
        try:
            war_freq_code = war_freq_encoder.transform([war_frequency])[0]
        except:
            war_freq_code = 0
        
        try:
            clan_type_code = clan_type_encoder.transform([clan_type])[0]
        except:
            clan_type_code = 0
        
        is_family_friendly_code = 1 if is_family_friendly else 0
        
        # 모델 입력
        X_input = np.array([[
            activity_ratio,
            entry_gap,
            war_freq_code,
            is_family_friendly_code,
            clan_type_code
        ]])
        
        # 예측
        survival_prob = survival_model.predict_proba(X_input)[0][1]
        
        # 결과 표시
        st.markdown("---")
        st.subheader("📊 진단 결과")
        
        if survival_prob >= 0.85:
            status = "🟢 안전"
            message = "이 클랜은 매우 안전합니다! 오래 유지될 가능성이 높습니다."
        elif survival_prob >= 0.6:
            status = "🟡 보통"
            message = "그럭저럭 안정적입니다. 활동성을 높이면 좀 더 좋아질 수 있어요."
        else:
            status = "🔴 위험"
            message = "이탈 위험이 있습니다! 클랜 관리에 신경 좀 쓰세요."
        
        st.metric(label="생존 확률", value=f"{survival_prob:.1%}", delta=status)
        st.markdown(f"### {message}")
        
        with st.expander("세부 분석 보기"):
            st.write(f"- **활동 효율성** (Activity Ratio): {activity_ratio:.2f}")
            st.write(f"- **진입 장벽 격차** (Entry Gap): {entry_gap:,}")
            if activity_ratio < 15:
                st.warning("⚠️ 활동 효율성이 낮습니다. 멤버들의 트로피 활동을 장려하세요!")
            if entry_gap < 500:
                st.warning("⚠️ 진입 장벽이 너무 낮습니다. 가입 조건을 조정해 보세요!")

# ==========================================
# 탭 2: 리그 등급 예측
# ==========================================
with tab2:
    st.subheader("🏆 리그 등급 예측기")
    st.markdown("클랜의 현재 상태로 어느 리그까지 올라갈 수 있을지 예측합니다")
    
    col1, col2 = st.columns(2)
    
    with col1:
        clan_level = st.number_input(
            "클랜 레벨",
            min_value=1, max_value=30, value=10,
            help="현재 클랜 레벨",
            key="league_clan_level"
        )
        
        clan_points = st.number_input(
            "클랜 포인트",
            min_value=0, max_value=100000, value=20000,
            help="클랜 총 포인트",
            key="league_clan_points"
        )
        
        war_wins = st.number_input(
            "클랜전 승리 수",
            min_value=0, max_value=2000, value=100,
            help="총 클랜전 승리 횟수",
            key="league_war_wins"
        )
        
        clan_capital_points = st.number_input(
            "클랜 캐피탈 포인트",
            min_value=0, max_value=100000, value=5000,
            help="클랜 캐피탈 총 포인트",
            key="league_capital_points"
        )
        
        mean_level = st.number_input(
            "멤버 평균 레벨",
            min_value=1, max_value=300, value=120,
            help="클랜원들의 평균 경험치 레벨",
            key="league_mean_level"
        )
    
    with col2:
        mean_trophies = st.number_input(
            "멤버 평균 트로피",
            min_value=0, max_value=6000, value=2000,
            help="클랜원들의 평균 트로피",
            key="league_mean_trophies"
        )
        
        activity_ratio_input = st.number_input(
            "활동성 지수",
            min_value=0.0, max_value=100.0, value=15.0,
            help="트로피 / (레벨 + 1)",
            key="league_activity"
        )
        
        entry_gap_input = st.number_input(
            "진입 장벽 격차",
            min_value=-5000, max_value=5000, value=500,
            help="평균 트로피 - 가입 조건 트로피",
            key="league_entry_gap"
        )
        
        points_per_member = st.number_input(
            "멤버당 포인트",
            min_value=0.0, max_value=5000.0, value=500.0,
            help="클랜 포인트 / 멤버 수",
            key="league_points_per_member"
        )
    
    if st.button("🔍 리그 등급 예측", type="primary", use_container_width=True, key="league_btn"):
        # 모델 입력 (9개 변수)
        X_input = np.array([[
            clan_level,
            clan_points,
            war_wins,
            clan_capital_points,
            mean_level,
            mean_trophies,
            activity_ratio_input,
            entry_gap_input,
            points_per_member
        ]])
        
        # 예측
        pred_encoded = league_model.predict(X_input)[0]
        pred_league = league_encoder.inverse_transform([pred_encoded])[0]
        
        # 확률 분포 (가능하면)
        try:
            proba = league_model.predict_proba(X_input)[0]
            classes = league_encoder.classes_
        except:
            proba = None
            classes = None
        
        # session_state에 결과 저장
        st.session_state['league_result'] = {
            'pred_league': pred_league,
            'proba': proba,
            'classes': classes,
            'input_values': {
                'clan_level': clan_level,
                'clan_points': clan_points,
                'war_wins': war_wins,
                'clan_capital_points': clan_capital_points,
                'mean_member_level': mean_level,
                'mean_member_trophies': mean_trophies,
                'activity_ratio': activity_ratio_input,
                'entry_gap': entry_gap_input,
                'points_per_member': points_per_member
            }
        }
    
    # session_state에 결과가 있으면 표시
    if 'league_result' in st.session_state:
        result = st.session_state['league_result']
        pred_league = result['pred_league']
        proba = result['proba']
        classes = result['classes']
        current_values = result['input_values']
        
        # 결과 표시
        st.markdown("---")
        st.subheader("📊 예측 결과")
        
        # 리그별 이모지
        league_emoji = {
            'Bronze': '🥉', 'Silver': '🥈', 'Gold': '🥇',
            'Crystal': '💎', 'Master': '🔥', 'Champion': '👑'
        }
        
        emoji = league_emoji.get(pred_league, '🏆')
        st.metric(label="예측 리그", value=f"{emoji} {pred_league}")
        
        # 확률 분포 표시
        if proba is not None:
            st.markdown("### 📈 리그별 확률 분포")
            
            # 순서대로 정렬
            tier_order = ['Bronze', 'Silver', 'Gold', 'Crystal', 'Master', 'Champion']
            sorted_data = []
            for tier in tier_order:
                if tier in classes:
                    idx = list(classes).index(tier)
                    sorted_data.append((tier, proba[idx]))
            
            for tier, prob in sorted_data:
                emoji_tier = league_emoji.get(tier, '')
                st.write(f"{emoji_tier} **{tier}**: {prob:.1%}")
        
        # ±1 티어 설명
        with st.expander("ℹ️ 예측 정확도 안내"):
            st.info("""
            **모델 정확도**: 약 65%  
            **±1 티어 허용 시**: 약 98%
            
            예를 들어 Gold로 예측했다면, 실제 리그가 Silver~Crystal 범위일 확률이 98%입니다!
            """)
        
        # ==========================================
        # 성장 가이드
        # ==========================================
        st.markdown("---")
        st.subheader("📈 성장 가이드")
        
        tier_order = ['Bronze', 'Silver', 'Gold', 'Crystal', 'Master', 'Champion']
        current_idx = tier_order.index(pred_league) if pred_league in tier_order else 0
        
        # 현재 티어보다 높은 티어만 선택 가능
        available_tiers = tier_order[current_idx + 1:] if current_idx < len(tier_order) - 1 else []
        
        if not available_tiers:
            st.success("🎉 축하합니다! 이미 최고 티어(Champion)입니다!")
        else:
            # 목표 티어 선택
            target_tier = st.selectbox(
                "🎯 목표 티어 선택",
                options=available_tiers,
                index=0,
                help="도달하고 싶은 목표 티어를 선택하세요",
                key="target_tier_select"
            )
            
            target_emoji = league_emoji.get(target_tier, '🏆')
            st.markdown(f"**현재 티어**: {league_emoji.get(pred_league, '')} {pred_league} → **목표 티어**: {target_emoji} {target_tier}")
            
            # 목표 티어 기준값
            if target_tier in tier_standards.index:
                target_standards = tier_standards.loc[target_tier]
                
                st.markdown("#### 🎯 개선이 필요한 항목")
                
                feature_names_ko = {
                    'clan_level': '클랜 레벨',
                    'clan_points': '클랜 포인트',
                    'war_wins': '클랜전 승리 수',
                    'clan_capital_points': '캐피탈 포인트',
                    'mean_member_level': '멤버 평균 레벨',
                    'mean_member_trophies': '멤버 평균 트로피',
                    'activity_ratio': '활동성 지수',
                    'entry_gap': '진입 장벽 격차',
                    'points_per_member': '멤버당 포인트'
                }
                
                improvements = []
                for feature, current in current_values.items():
                    if feature in target_standards.index:
                        target = target_standards[feature]
                        diff = target - current
                        if diff > 0:
                            improvements.append({
                                'feature': feature_names_ko.get(feature, feature),
                                'current': current,
                                'target': target,
                                'diff': diff
                            })
                
                if improvements:
                    # 중요도 순으로 정렬 (diff 크기 기준)
                    improvements.sort(key=lambda x: x['diff'], reverse=True)
                    
                    for item in improvements[:5]:  # 상위 5개만 표시
                        if item['diff'] > 0.01:  # 미미한 차이는 제외
                            st.write(f"- **{item['feature']}**: 현재 {item['current']:,.1f} → 목표 {item['target']:,.1f} (📈 +{item['diff']:,.1f})")
                else:
                    st.success(f"👍 모든 수치가 {target_tier} 티어 기준을 충족합니다! 조금만 더 노력하세요!")
            else:
                st.warning("티어 기준 데이터를 찾을 수 없습니다.")

# ==========================================
# 푸터
# ==========================================
st.markdown("---")
st.caption("Made with ❤️ by ML Team | Data: Clash of Clans API")
