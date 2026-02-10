import streamlit as st


def show_learning_modules():

    st.markdown("""
        <style>
            .module-card {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 16px;
                padding: 30px;
                margin-top: 40px;
                margin-bottom: 30px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.1);
                transition: 0.3s ease-in-out;
            }
            .module-card:hover {
                box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            }
            .module-title {
                font-size: 22px;
                font-weight: bold;
                color: #3A7CA5;
                margin-bottom: 10px;
            }
            .module-outcome {
                font-size: 15px;
                color: #444;
                margin-bottom: 20px;
                line-height: 1.6;
            }
            .learn-btn {
                background-color: #3A7CA5;
                color: white;
                border: none;
                padding: 7px 18px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                transition: 0.3s;
                margin-bottom: 1rem;
            }
            .learn-btn:hover {
                background-color: #245d7a;
            }
        </style>
    """, unsafe_allow_html=True)

    # Page headers
    st.markdown('<div class="main-title">📚 Learning Modules</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Learn, practice, and master your speaking skills</div>', unsafe_allow_html=True)

    # Module data
    modules = [
        {
            "title": "Module 1 - Posture & Presence",
            "outcome": "Understand how posture influences audience perception and develop confident, balanced body language that projects credibility during speeches.",
            "subs": [
                ("1.1", "Importance of Posture in Public Speaking", "https://youtu.be/Ks-_Mh1QhMc?si=bh4_4B-nnpS1puhX"),
                ("1.2", "Common Posture Mistakes and How to Fix Them", "https://youtu.be/KP7XdGmtX5I"),
                ("1.3", "Exercises to Improve Body Alignment and Balance", "https://youtu.be/mS_VU71kS24?si=1AxzFrrZPXRKILTh"),
                ("1.4", "Applying Confident Posture During Real Speeches", "https://youtu.be/U1O3UFeCEeU?si=YS777AE_CYr81LtR"),
            ],
        },
        {
            "title": "Module 2 - Eye Contact Mastery",
            "outcome": "Build natural, controlled eye-contact habits that strengthen audience connection and trust in both in-person and virtual settings.",
            "subs": [
                ("2.1", "Why Eye Contact Builds Audience Trust?", "https://youtu.be/MGhwnXV80ZQ?si=T8eyO8iyZqnWaHqS"),
                ("2.2", "Maintaining Natural Eye Movement Without Staring", "https://youtu.be/Ior3iIXjMTU?si=eJTq2FvxEOBJNmnk"),
                ("2.3", "Techniques for Handling Large or Virtual Audiences", "https://youtu.be/JtTl8-TsZpk?si=lW-OgwIpEl7nYeUK"),
                ("2.4", "Practicing Eye Contact with a Mirror or Camera", "https://youtu.be/8OGDhlUvSK4?si=4qvqnGRgtXlkpZzI"),
            ],
        },
        {
            "title": "Module 3 - Tone & Confidence",
            "outcome": "Use vocal tone, variety, and breathing techniques to convey confidence, reduce anxiety, and emotionally engage listeners.",
            "subs": [
                ("3.1", "How Tone Reflects Your Confidence and Emotion", "https://youtu.be/hPQyHXc1ksA?si=EGevfjyA8lkZs5V1"),
                ("3.2", "Avoiding Monotone Delivery — Adding Vocal Variety", "https://youtu.be/ikuNOsPU5E0?si=VjquxImV5-J5E9Jv"),
                ("3.3", "Voice Warm-Up & Breathing Exercises for Power", "https://youtu.be/7eDcHZZn7hU?si=Hz0dFLhz5gDsx71a"),
                ("3.4", "Managing Stage Anxiety and Nervousness", "https://youtu.be/VEStYVONy-0?si=M_Q4C4AIfVmpGTbz"),
            ],
        },
        {
            "title": "Module 4 - Speech Speed & Pauses",
            "outcome": "Master pacing, rhythm, and pauses to deliver impactful speeches that maintain audience attention.",
            "subs": [
                ("4.1", "Finding Your Ideal Speaking Speed", "https://youtu.be/032Hum9KNjw?si=QNP8UZzErO90lLJG"),
                ("4.2", "Using Strategic Pauses for Impact", "https://youtu.be/kXsd25D25HI?si=oucAnC9kM9FkKvwr"),
                ("4.3", "Exercises to Control Speed and Breathing", "https://youtu.be/QxpR2_gwUEY?si=rXj91M0YWZ42Px4-"),
                ("4.4", "Improving Clarity and Rhythm While Speaking", "https://youtu.be/5YtbvUSdt5Q?si=ck6OfVIeJfUix3FE"),
            ],
        },
    ]


    for module in modules:

        st.markdown(
            f"<div class='module-card'><div class='module-title'>{module['title']}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='module-outcome'>{module['outcome']}</div>",
            unsafe_allow_html=True
        )

        first = module["subs"][0]

        st.markdown(f"**{first[0]} {first[1]}**")

        st.markdown(f"""
            <a href="{first[2]}" target="_blank">
                <button class="learn-btn">Learn ▶</button>
            </a>
        """, unsafe_allow_html=True)


        with st.expander("Show more submodules"):

            for sub in module["subs"][1:]:

                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(f"- {sub[0]} {sub[1]}")

                with col2:
                    st.markdown(f"""
                        <a href="{sub[2]}" target="_blank">
                            <button class="learn-btn">Learn ▶</button>
                        </a>
                    """, unsafe_allow_html=True)


        st.markdown("</div>", unsafe_allow_html=True)
