"""
🎮 클랜 건강 체크기 (Clan Health Checker)
Clash of Clans 클랜의 앞으로의 생존 확률을 예측하고 방향성을 제시해주는 웹 앱

실행 방법: streamlit run app.py
"""
import streamlit as st
import joblib
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="클랜 생존 예측기",
    page_icon="⚔️",
    layout="centered"
)

# 모델 및 인코더 로드
@st.cache_resource
def load_models():
    model = joblib.load('clan_retention_model.pkl')
    war_freq_encoder = joblib.load('war_frequency_encoder.pkl')
    clan_type_encoder = joblib.load('clan_type_encoder.pkl')
    return model, war_freq_encoder, clan_type_encoder

model, war_freq_encoder, clan_type_encoder = load_models()

# 헤더
st.title("클랜 생존 예측기")
st.markdown("**당신의 클랜은 앞으로도 살아남을 수 있을까요?**")
st.markdown("---")

# 입력 폼
st.subheader("클랜 정보 입력")

col1, col2 = st.columns(2)

with col1:
    mean_member_trophies = st.number_input(
        "멤버 평균 트로피",
        min_value=0, max_value=6000, value=1500,
        help="클랜원들의 평균 트로피 점수"
    )
    
    mean_member_level = st.number_input(
        "멤버 평균 레벨 ",
        min_value=1, max_value=300, value=100,
        help="클랜원들의 평균 경험치 레벨"
    )
    
    required_trophies = st.number_input(
        "가입 조건 트로피",
        min_value=0, max_value=5500, value=800,
        help="클랜 가입에 필요한 최소 트로피"
    )

with col2:
    war_frequency = st.selectbox(
        "전쟁 빈도 설정",
        options=['always', 'moreThanOncePerWeek', 'oncePerWeek', 'lessThanOncePerWeek', 'never', 'unknown'],
        index=0,
        help="클랜의 전쟁 빈도 설정값"
    )
    
    clan_type = st.selectbox(
        "클랜 공개 설정 ",
        options=['inviteOnly', 'open', 'closed'],
        index=0,
        help="클랜의 가입 방식"
    )
    
    is_family_friendly = st.checkbox(
        "가족 친화 모드 ",
        value=True,
        help="가족 친화 설정 여부"
    )

st.markdown("---")

# 예측 버튼
if st.button("생존 확률 확인", type="primary", use_container_width=True):
    
    # 1. 파생변수 계산
    activity_ratio = mean_member_trophies / (mean_member_level + 1)
    entry_gap = mean_member_trophies - required_trophies
    
    # 2. 인코딩
    try:
        war_freq_code = war_freq_encoder.transform([war_frequency])[0]
    except:
        war_freq_code = 0  # 알 수 없는 값이면 기본값
    
    try:
        clan_type_code = clan_type_encoder.transform([clan_type])[0]
    except:
        clan_type_code = 0
    
    is_family_friendly_code = 1 if is_family_friendly else 0
    
    # 3. 모델 입력 준비 (순서 중요!)
    # engineered_features_v2 = ['activity_ratio', 'entry_gap', 'war_frequency_code', 'isFamilyFriendly', 'clan_type_code']
    X_input = np.array([[
        activity_ratio,
        entry_gap,
        war_freq_code,
        is_family_friendly_code,
        clan_type_code
    ]])
    
    # 4. 예측
    survival_prob = model.predict_proba(X_input)[0][1]
    
    # 5. 결과 표시
    st.markdown("---")
    st.subheader(" 진단 결과")
    
    # 점수에 따른 색상 및 메시지
    if survival_prob >= 0.85:
        color = "green"
        status = "🟢 안전"
        message = "이 클랜은 매우 안전합니다! 오래 유지될 가능성이 높습니다."
    elif survival_prob >= 0.6:
        color = "orange"
        status = "🟡 보통"
        message = "그럭저럭 안정적입니다. 활동성을 높이면 좀 더 좋아질 수 있어요."
    else:
        color = "red"
        status = "🔴 위험"
        message = "이탈 위험이 있습니다! 클랜 관리에 신경 좀 쓰세요."
    
    # 큰 숫자로 표시
    st.metric(
        label="생존 확률",
        value=f"{survival_prob:.1%}",
        delta=status
    )
    
    st.markdown(f"### {message}")
    
    # 세부 분석
    with st.expander("세부 분석 보기"):
        st.write(f"- **활동 효율성** (Activity Ratio): {activity_ratio:.2f}")
        st.write(f"- **진입 장벽 격차** (Entry Gap): {entry_gap:,}")
        st.write(f"- **전쟁 빈도 코드**: {war_freq_code}")
        st.write(f"- **클랜 유형 코드**: {clan_type_code}")
        
        if activity_ratio < 15:
            st.warning(" 활동 효율성이 낮습니다. 멤버들의 트로피 활동을 장려하세요!")
        if entry_gap < 500:
            st.warning(" 진입 장벽이 너무 낮습니다. 가입 조건을 조정해 보세요!")

# 푸터
st.markdown("---")
st.caption("Made with by ML Team | Data: Clash of Clans API")
