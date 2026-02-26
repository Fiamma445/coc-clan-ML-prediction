"""
🎮 클랜 성장 코칭 (Clan Growth Coaching)
클랜의 예상 리그 티어를 예측하고, 성장을 위한 개선점을 제안합니다.

실행 방법: uv run streamlit run app_coaching.py
"""
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="클랜 성장 코칭",
    page_icon="📈",
    layout="centered"
)

# 모델 로드
@st.cache_resource
def load_models():
    try:
        model = joblib.load('clan_league_model.pkl')
        return model
    except FileNotFoundError:
        st.error("⚠️ 모델 파일(clan_league_model.pkl)을 찾을 수 없습니다. 노트북에서 먼저 저장해주세요.")
        st.code("joblib.dump(model_cl1, 'clan_league_model.pkl')", language="python")
        return None

# 티어별 평균값 (실제 데이터 기반)
TIER_STANDARDS = {
    0: {'clan_level': 1.58, 'clan_points': 3825, 'clan_capital_points': 54, 'num_members': 8, 'required_townhall_level': 2, 'required_trophies': 249, 'mean_member_level': 52},
    1: {'clan_level': 3.16, 'clan_points': 6013, 'clan_capital_points': 80, 'num_members': 16, 'required_townhall_level': 3, 'required_trophies': 334, 'mean_member_level': 46},
    2: {'clan_level': 5.52, 'clan_points': 11841, 'clan_capital_points': 350, 'num_members': 23, 'required_townhall_level': 5, 'required_trophies': 603, 'mean_member_level': 78},
    3: {'clan_level': 9.30, 'clan_points': 19303, 'clan_capital_points': 857, 'num_members': 30, 'required_townhall_level': 7, 'required_trophies': 1047, 'mean_member_level': 109},
    4: {'clan_level': 15.89, 'clan_points': 30457, 'clan_capital_points': 1774, 'num_members': 38, 'required_townhall_level': 10, 'required_trophies': 1670, 'mean_member_level': 150},
    5: {'clan_level': 21.03, 'clan_points': 38153, 'clan_capital_points': 2615, 'num_members': 39, 'required_townhall_level': 12, 'required_trophies': 2121, 'mean_member_level': 190},
    6: {'clan_level': 22.43, 'clan_points': 36027, 'clan_capital_points': 3226, 'num_members': 32, 'required_townhall_level': 12, 'required_trophies': 2248, 'mean_member_level': 203},
}

TIER_NAMES = {
    0: "언랭크 (Unranked)",
    1: "브론즈 (Bronze)", 
    2: "실버 (Silver)",
    3: "골드 (Gold)",
    4: "크리스탈 (Crystal)",
    5: "마스터 (Master)",
    6: "챔피언 (Champion)"
}

FEATURE_NAMES_KR = {
    'clan_level': '클랜 레벨',
    'clan_points': '클랜 점수',
    'clan_capital_points': '캐피탈 점수',
    'num_members': '멤버 수',
    'required_townhall_level': '가입 타운홀 제한',
    'required_trophies': '가입 트로피 조건',
    'mean_member_level': '멤버 평균 레벨'
}

model = load_models()

# 헤더
st.title("📈 클랜 성장 코칭")
st.markdown("**당신의 클랜은 어떤 리그에 속해있어야 할까요? 성장 포인트를 알려드립니다!**")
st.markdown("---")

# 입력 폼
st.subheader("📝 클랜 정보 입력")

col1, col2 = st.columns(2)

with col1:
    clan_level = st.number_input(
        "클랜 레벨 🏰",
        min_value=1, max_value=30, value=10,
        help="클랜의 현재 레벨"
    )
    
    clan_points = st.number_input(
        "클랜 점수 🏆",
        min_value=0, max_value=100000, value=20000,
        help="클랜의 총 점수"
    )
    
    clan_capital_points = st.number_input(
        "캐피탈 점수 🏛️",
        min_value=0, max_value=50000, value=3000,
        help="클랜 캐피탈 점수"
    )
    
    num_members = st.slider(
        "멤버 수 👥",
        min_value=1, max_value=50, value=25,
        help="현재 클랜원 수"
    )

with col2:
    required_townhall_level = st.number_input(
        "가입 타운홀 제한 🏠",
        min_value=1, max_value=16, value=8,
        help="가입에 필요한 최소 타운홀 레벨"
    )
    
    required_trophies = st.number_input(
        "가입 트로피 조건 🏅",
        min_value=0, max_value=5500, value=1000,
        help="가입에 필요한 최소 트로피"
    )
    
    mean_member_level = st.number_input(
        "멤버 평균 레벨 📊",
        min_value=1, max_value=300, value=100,
        help="클랜원들의 평균 경험치 레벨"
    )
    
    target_tier = st.selectbox(
        "목표 티어 🎯 (선택사항)",
        options=[None, 0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: "자동 (예측 기반)" if x is None else TIER_NAMES.get(x, str(x)),
        help="달성하고 싶은 리그 티어를 선택하세요"
    )

st.markdown("---")

# 예측 버튼
if st.button("🔮 성장 코칭 받기", type="primary", use_container_width=True):
    
    if model is None:
        st.error("모델이 로드되지 않았습니다.")
    else:
        # 입력 데이터 준비
        features = ['clan_level', 'clan_points', 'clan_capital_points', 'num_members', 
                   'required_townhall_level', 'required_trophies', 'mean_member_level']
        
        X_input = np.array([[
            clan_level,
            clan_points,
            clan_capital_points,
            num_members,
            required_townhall_level,
            required_trophies,
            mean_member_level
        ]])
        
        # 예측
        predicted_tier = int(model.predict(X_input)[0])
        
        # 예측 확률 (있는 경우)
        try:
            proba = model.predict_proba(X_input)[0]
            confidence = proba[predicted_tier] * 100
        except:
            confidence = None
        
        # 결과 표시
        st.markdown("---")
        st.subheader("📊 진단 결과")
        
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            st.metric(
                label="예측 리그",
                value=TIER_NAMES.get(predicted_tier, f"Tier {predicted_tier}")
            )
        
        with col_result2:
            if confidence:
                st.metric(
                    label="예측 확신도",
                    value=f"{confidence:.1f}%"
                )
        
        # 목표 티어 설정
        if target_tier is None:
            # 자동: 예측 티어보다 1단계 위를 목표로
            goal_tier = min(predicted_tier + 1, 6)
        else:
            goal_tier = target_tier
        
        # 개선점 분석
        if goal_tier > predicted_tier:
            st.markdown("---")
            st.subheader(f"🚀 {TIER_NAMES.get(goal_tier, f'Tier {goal_tier}')} 달성을 위한 개선점")
            
            current_values = {
                'clan_level': clan_level,
                'clan_points': clan_points,
                'clan_capital_points': clan_capital_points,
                'num_members': num_members,
                'required_townhall_level': required_townhall_level,
                'required_trophies': required_trophies,
                'mean_member_level': mean_member_level
            }
            
            goal_standards = TIER_STANDARDS.get(goal_tier, TIER_STANDARDS[4])
            
            improvements = []
            for feature, current in current_values.items():
                target = goal_standards[feature]
                gap = target - current
                if gap > 0:  # 부족한 항목만
                    gap_pct = (gap / target) * 100 if target > 0 else 0
                    improvements.append({
                        'feature': feature,
                        'feature_kr': FEATURE_NAMES_KR[feature],
                        'current': current,
                        'target': target,
                        'gap': gap,
                        'gap_pct': gap_pct
                    })
            
            # 개선폭이 큰 순으로 정렬
            improvements.sort(key=lambda x: x['gap_pct'], reverse=True)
            
            if improvements:
                for i, item in enumerate(improvements[:5], 1):
                    with st.container():
                        col_a, col_b, col_c = st.columns([2, 1, 1])
                        with col_a:
                            st.write(f"**{i}. {item['feature_kr']}**")
                        with col_b:
                            st.write(f"현재: {item['current']:,}")
                        with col_c:
                            if item['gap'] > 0:
                                st.write(f"목표: {item['target']:,} (+{item['gap']:,.0f})")
                            else:
                                st.write(f"✅ 달성!")
                    
                    # 진행 바
                    progress = min(item['current'] / item['target'], 1.0) if item['target'] > 0 else 1.0
                    st.progress(progress)
                    st.write("")
            else:
                st.success("🎉 축하합니다! 이미 목표 티어의 기준을 모두 충족했습니다!")
        
        elif goal_tier == predicted_tier:
            st.success(f"✅ 현재 클랜 상태가 **{TIER_NAMES.get(predicted_tier)}** 수준에 적합합니다!")
        
        else:
            st.info(f"🏆 이미 목표 티어({TIER_NAMES.get(goal_tier)})보다 높은 수준입니다!")

# 푸터
st.markdown("---")
st.caption("Made with ❤️ by ML Team | Clan Growth Coaching System")
