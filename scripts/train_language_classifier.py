"""Train Naive Bayes Language Classifier with character n-grams (1-3).

Trains on authentic multilingual corpus covering Latin, Devanagari (Hindi, Sanskrit, Marathi),
Dravidian (Kannada, Tamil, Telugu), and Arabic scripts. Evaluates on a held-out test split
and outputs real measured accuracy.
"""

from __future__ import annotations

from pathlib import Path
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# Curated representative sentences and passages per language
CORPUS_DATA = {
    "en": [
        "Ancient manuscripts preserve the knowledge and philosophy of civilizations across centuries.",
        "The historical archive contains rare codices written by medieval scholars.",
        "Translation of classical treatises requires deep understanding of linguistics and cultural history.",
        "Paleography and codicology are essential disciplines for studying handwritten documents.",
        "Scribes illuminated the parchment manuscripts with gold leaf and vibrant pigments.",
        "The digital preservation of cultural heritage ensures knowledge reaches future generations.",
        "Linguistic analysis reveals historical connections between ancient European and Asian languages.",
        "Monastic libraries protected philosophical and scientific manuscripts through the Middle Ages.",
        "Scholars gathered at the ancient library of Alexandria to compile universal wisdom.",
        "Deciphering damaged papyri requires advanced multi-spectral imaging and neural networks.",
        "The declaration was signed on parchment and preserved in the national archives.",
        "Literary works from antiquity reflect the moral and artistic ideals of their creators.",
        "The Renaissance marked a profound revival of classical literature and humanist education.",
        "Archaeologists excavated parchment fragments dating back to the third century.",
        "Systematic cataloging of palimpsests reveals hidden texts beneath later writings.",
    ],
    "hi": [
        "यह एक प्राचीन पांडुलिपि का नमूना है जो भारतीय इतिहास को दर्शाता है।",
        "पुरातन ग्रंथों में भारतीय दर्शन और विज्ञान का गहरा ज्ञान समाहित है।",
        "इतिहासकारों ने मध्यकालीन भारत के दुर्लभ दस्तावेजों का संकलन किया है।",
        "भाषा विज्ञान की दृष्टि से हिंदी एक समृद्ध और जीवंत भाषा है।",
        "प्राचीन नालंदा विश्वविद्यालय में हस्तलिखित पांडुलिपियों का विशाल संग्रह था।",
        "भारतीय संस्कृति में साहित्य और कला का अद्वितीय स्थान रहा है।",
        "अध्ययन और अनुसंधान से हमें पूर्वजों के ज्ञान और उपलब्धियों का बोध होता है।",
        "सांस्कृतिक विरासत का संरक्षण हमारे समाज का परम कर्तव्य है।",
        "मुंशी प्रेमचंद के उपन्यासों में ग्रामीण समाज का यथार्थ चित्रण मिलता है।",
        "कालिदास के नाटकों का अनुवाद संसार की अनेक प्रमुख भाषाओं में हुआ है।",
        "आधुनिक युग में प्राचीन ग्रंथों का डिजिटलीकरण एक सराहनीय कदम है।",
        "भारत के विभिन्न राज्यों में पांडुलिपि संरक्षण केंद्र स्थापित किए गए हैं।",
        "ज्ञान और विज्ञान की यह परंपरा सदियों से निरंतर प्रवाहित हो रही है।",
        "संस्कृत और प्राकृत साहित्य ने हिंदी भाषा के विकास में महत्वपूर्ण योगदान दिया।",
        "इतिहास के पृष्ठों में छिपे रहस्यों को उजागर करना विद्वानों का लक्ष्य है।",
        "ज्ञान और शिक्षा से ही किसी भी राष्ट्र का सर्वांगीण विकास संभव है।",
        "भारत की सांस्कृतिक विविधता में एकता ही इसकी सबसे बड़ी ताकत और पहचान है।",
        "आधुनिक युग में सूचना प्रौद्योगिकी ने मानव जीवन को बहुत सरल बना दिया है।",
        "साहित्य समाज का दर्पण होता है और मानवीय भावनाओं को अभिव्यक्त करता है।",
        "प्रकृति का संरक्षण हम सभी का नैतिक और मानवीय कर्तव्य है।",
    ],
    "sa": [
        "सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः सर्वे भद्राणि पश्यन्तु मा कश्चिद्दुःखभाग्भवेत् ।",
        "विद्या ददाति विनयं विनयाद्याति पात्रताम् पात्रत्वाद्धनमाप्नोति धनाद्धर्मं ततः सुखम् ।",
        "सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः येनाक्रमन्त्यृषयो ह्याप्तकामाः ।",
        "उद्यमेन हि सिध्यन्ति कार्याणि न मनोरथैः न हि सुप्तस्य सिंहस्य प्रविशन्ति मुखे मृगाः ।",
        "वसुधैव कुटुम्बकम् इति उदारचरितानां तु वसुधैव कुटुम्बकम् भवति ।",
        "अहिंसा परमो धर्मः धर्महिंसा तथैव च । यतो धर्मस्ततो जयः सर्वदा विजयते ।",
        "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम् ।",
        "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ।",
        "गुरुर्ब्रह्मा गुरुर्विष्णुः गुरुर्देवो महेश्वरः गुरुः साक्षात्परं ब्रह्म तस्मै श्रीगुरवे नमः ।",
        "ज्ञानेन हीनाः पशुभिः समानाः भवन्ति इति शास्त्रेषु प्रतिपादितम् ।",
        "अष्टादशपुराणेषु व्यासस्य वचनद्वयम् परोपकारः पुण्याय पापाय परपीडनम् ।",
        "ऋग्वेदः यजुर्वेदः सामवेदः अथर्ववेदश्चेति चत्वारो वेदाः प्राचीनाः सन्ति ।",
        "काव्यशास्त्रविनोदेन कालो गच्छति धीमताम् व्यसनेन तु मूर्खाणां निद्रया कलहेन वा ।",
        "मातृदेवो भव पितृदेवो भव आचार्यदेवो भव अतिथिदेवो भव ।",
        "न हि ज्ञानेन सदृशं पवित्रमिह विद्यते तत्स्वयं योगसंसिद्धः कालेनात्मनि विन्दति ।",
        "सत्यं वद धर्मं चर स्वाध्यायान्मा प्रमदः आचार्याय प्रियं धनमाहृत्य प्रजातन्तुं मा व्यवच्छेत्सीः ।",
        "अहिंसा सत्यमस्तेयं शौचमिन्द्रियनिग्रहः एतं सामासिकं धर्मं चातुर्वर्ण्येऽब्रवीन्मनुः ।",
        "विद्वान् सर्वत्र पूज्यते राजा स्वदेशे पूज्यते विद्वत्त्वं च नृपत्वं च नैव तुल्यं कदाचन ।",
        "परोपकारार्थमिदं शरीरं परोपकाराय फलन्ति वृक्षाः परोपकाराय वहन्ति नद्यः ।",
        "धर्मो रक्षति रक्षितः तस्माद्धर्मो न हन्तव्यो मा नो धर्मो हतोऽवधीत् ।",
    ],
    "kn": [
        "ಪ್ರಾಚೀನ ಹಸ್ತಪ್ರತಿಗಳು ಭಾರತದ ಇತಿಹಾಸ ಮತ್ತು ಪರಂಪರೆಯನ್ನು ತಿಳಿಸುತ್ತವೆ.",
        "ಕನ್ನಡ ಸಾಹಿತ್ಯವು ಸಾವಿರಾರು ವರ್ಷಗಳ ಸಮೃದ್ಧ ಇತಿಹಾಸವನ್ನು ಹೊಂದಿದೆ.",
        "ಕವಿರಾಜಮಾರ್ಗವು ಕನ್ನಡದಲ್ಲಿ ದೊರೆತಿರುವ ಅತ್ಯಂತ ಪ್ರಾಚೀನ ಗ್ರಂಥವಾಗಿದೆ.",
        "ಕರ್ನಾಟಕದ ದೇವಾಲಯಗಳ ಶಿಲಾಶಾಸನಗಳಲ್ಲಿ ಭಾಷೆಯ ಉಗಮವನ್ನು ಕಾಣಬಹುದು.",
        "ಪಂಪ, ರನ್ನ, ಜನ್ನ ಮೊದಲಾದ ಪ್ರಾಚೀನ ಕವಿಗಳು ಕನ್ನಡ ಸಾಹಿತ್ಯವನ್ನು ಶ್ರೀಮಂತಗೊಳಿಸಿದ್ದಾರೆ.",
        "ವಚನ ಸಾಹಿತ್ಯವು ಸಮಾಜ ಸುಧಾರಣೆಗೆ ಮಹತ್ತರವಾದ ಕೊಡುಗೆಯನ್ನು ನೀಡಿದೆ.",
        "ಬಸವೇಶ್ವರರು ತಮ್ಮ ವಚನಗಳ ಮೂಲಕ ಸಾಮಾಜಿಕ ಸಮಾನತೆಯನ್ನು ಸಾರಿದರು.",
        "ತಾಳೆಗರಿ ಹಸ್ತಪ್ರತಿಗಳನ್ನು ಸಂರಕ್ಷಿಸುವುದು ಇಂದಿನ ಪೀಳಿಗೆಯ ಕರ್ತವ್ಯವಾಗಿದೆ.",
        "ಜ್ಞಾನಪೀಠ ಪ್ರಶಸ್ತಿ ಪುರಸ್ಕೃತ ಕನ್ನಡ ಸಾಹಿತಿಗಳು ನಾಡಿನ ಹೆಮ್ಮೆಯಾಗಿದ್ದಾರೆ.",
        "ಕನ್ನಡ ಭಾಷೆಯು ಭಾರತದ ಅತ್ಯಂತ ಪ್ರಾಚೀನ ಶಾಸ್ತ್ರೀಯ ಭಾಷೆಗಳಲ್ಲಿ ಒಂದಾಗಿದೆ.",
        "ಮೈಸೂರು ವಿಶ್ವವಿದ್ಯಾನಿಲಯದ ಹಸ್ತಪ್ರತಿ ಭಂಡಾರದಲ್ಲಿ ಅಮೂಲ್ಯ ಕೃತಿಗಳಿವೆ.",
        "ನಾಡು, ನುಡಿ ಮತ್ತು ಸಂಸ್ಕೃತಿಯ ರಕ್ಷಣೆ ಪ್ರತಿಯೊಬ್ಬ ಕನ್ನಡಿಗನ ಜವಾಬ್ದಾರಿ.",
        "ಕಲೆಯು ಮಾನವನ ಜೀವನಕ್ಕೆ ಶಾಂತಿ ಮತ್ತು ಆನಂದವನ್ನು ನೀಡುತ್ತದೆ.",
        "ಐತಿಹಾಸಿಕ ಸ್ಮಾರಕಗಳು ನಮ್ಮ ಪೂರ್ವಜರ ವಾಸ್ತುಶಿಲ್ಪದ ಪ್ರತಿಭೆಯನ್ನು ಪ್ರತಿಬಿಂಬಿಸುತ್ತವೆ.",
        "ಶಾಸ್ತ್ರೀಯ ಕೃತಿಗಳ ಅಧ್ಯಯನವು ಭಾಷಾ ಪ್ರೌಢಿಮೆಯನ್ನು ಹೆಚ್ಚಿಸುತ್ತದೆ.",
        "ಕನ್ನಡ ನಾಡು ಸುಂದರವಾದ ನಾಡು ಮತ್ತು ಅತ್ಯಂತ ಶ್ರೀಮಂತ ನೈಸರ್ಗಿಕ ಸಂಪತ್ತನ್ನು ಹೊಂದಿದೆ.",
        "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿಯಾಗಿದ್ದು ತಂತ್ರಜ್ಞಾನ ಮತ್ತು ನಾವೀನ್ಯತೆಯ ಪ್ರಮುಖ ಕೇಂದ್ರವಾಗಿದೆ.",
        "ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ ರಚಿತವಾದ ಸಾಹಿತ್ಯ ಕೃತಿಗಳು ವಿಶ್ವಮಾನ್ಯತೆ ಮತ್ತು ಗೌರವವನ್ನು ಪಡೆದಿವೆ.",
        "ಪ್ರತಿಯೊಬ್ಬ ನಾಗರಿಕನೂ ತನ್ನ ನಾಡಿನ ಪರಂಪರೆ, ಸಂಸ್ಕೃತಿ ಮತ್ತು ನಿಸರ್ಗವನ್ನು ಪ್ರೀತಿಸಬೇಕು.",
        "ಜ್ಞಾನ ಮತ್ತು ಶಿಕ್ಷಣದಿಂದ ಮಾತ್ರ ಸಮಾಜದಲ್ಲಿ ಸಕಾರಾತ್ಮಕ ಬದಲಾವಣೆ ತರಲು ಸಾಧ್ಯ.",
    ],
    "ta": [
        "பண்டைய ஓலைச்சுவடிகள் தமிழர்களின் கலாச்சாரம் மற்றும் அறிவியல் அறிவை விளக்குகின்றன.",
        "தொல்காப்பியம் தமிழ் மொழியின் மிகத் தொன்மையான இலக்கண நூலாகும்.",
        "திருக்குறள் உலகப் பொதுமறையாகப் போற்றப்படும் உன்னத நீதி இலக்கியமாகும்.",
        "சங்க இலக்கியங்கள் தமிழர்களின் வீரம், காதல் மற்றும் இயற்கை அழகை விவரிக்கின்றன.",
        "கம்பராமாயணம் தமிழ் காவிய வரலாற்றில் ஒரு தனித்துவமான காவியமாகும்.",
        "தமிழ் மொழி உலக செம்மொழிகளில் முதன்மையானதும் தொன்மையானதும் ஆகும்.",
        "தஞ்சைப் பெரிய கோவில் தமிழ் கட்டடக் கலையின் உன்னத சிகரமாகும்.",
        "பாரதியாரின் கவிதைகள் விடுதலை உணர்வையும் பெண் விடுதலையையும் போற்றின.",
        "ஓலைச்சுவடி பாதுகாப்பு நடுவங்கள் அரிய தமிழ்ப் பொக்கிஷங்களைப் பாதுகாக்கின்றன.",
        "அறம், பொருள், இன்பம், வீடு என்ற நான்கையும் தமிழ் இலக்கியம் விளக்குகிறது.",
        "தமிழ்நாடு வரலாற்று ஆவணக் காப்பகத்தில் பல்லாயிரக்கணக்கான சுவடிகள் உள்ளன.",
        "பக்தி இலக்கியக் காலம் தமிழ் மொழியின் இசை மற்றும் பக்தி வளத்தை வளர்த்தது.",
        "சிலப்பதிகாரமும் மணிமேகலையும் இரட்டைக் காப்பியங்களாகப் புகழ்பெற்றவை.",
        "அறிவியல் தமிழின் வளர்ச்சி இன்றைய இணையதள யுகத்தில் மிக முக்கியமானது.",
        "பண்டைக்கால வணிகத் தொடர்புகள் தமிழ் மொழியின் உலகளாவிய செல்வாக்கைக் காட்டுகின்றன.",
    ],
    "te": [
        "ప్రాచీన తాళపత్ర గ్రంథాలు భారతీయ విజ్ఞాన సంపదను మనకు అందిస్తున్నాయి.",
        "తెలుగు భాష అత్యంత మధురమైన ప్రాచీన దేశభాషగా గుర్తింపు పొందింది.",
        "ఆంధ్ర మహాభారతమును కవిత్రయం నన్నయ, తిక్కన, ఎర్రన రమణీయంగా రచించారు.",
        "పోతన రచించిన శ్రీమదాంధ్ర భాగవతము భక్తి రస ప్రధానమైన మహాగ్రంథము.",
        "విజయ నగర సామ్రాజ్య కాలంలో శ్రీకృష్ణదేవరాయలు సాహిత్యాన్ని విశేషంగా పోషించారు.",
        "తెలుగు లిపి అత్యంత సుందరమైన గుండ్రటి అక్షరాలతో ప్రసిద్ధి చెందింది.",
        "తెలుగు సాహిత్యంలో శతక ప్రక్రియ ప్రజలలో నైతిక విలువలను పెంపొందించింది.",
        "ప్రాచీన శాసన పరిశోధనలు తెలుగు భాష ప్రాచీనతను తిరుగులేకుండా నిరూపించాయి.",
        "త్యాగరాజు కీర్తనలు కర్ణాటక సంగీత ప్రపంచంలో అత్యున్నత స్థానాన్ని పొందాయి.",
        "హస్తప్రతుల పరిరక్షణకు ప్రత్యేక డిజిటల్ వ్యవస్థలు ఏర్పాటు చేయబడుతున్నాయి.",
        "గురజాడ అప్పారావు గారి రచనలు ఆధునిక తెలుగు సమాజానికి మార్గదర్శనం చేశాయి.",
        "వేమన పద్యాలు లోకనీతిని అత్యంత సరళమైన రీతిలో బోధిస్తాయి.",
        "తెలుగు భాషా సంస్కృతుల పరిరక్షణ మనందరి ప్రాథమిక బాధ్యత.",
        "సంస్కృత ప్రాకృత పదాల కలయికతో తెలుగు భాష ఎంతో సుసంపన్నమైనది.",
        "చారిత్రక గ్రంథాల పునరుద్ధరణ నూతన పరిశోధనలకు దోహదపడుతుంది.",
    ],
    "ar": [
        "تحتفظ المخطوطات القديمة بكنوز الفكر والعلوم والحضارة عبر العصور.",
        "يعتبر الخط العربي فناً إسلامياً فريداً يعكس جمال اللغة وعمقها.",
        "جمعت دار الحكمة في بغداد روائع التراث الفلسفي والعلمي في العصر العباسي.",
        "قام العلماء العرب والمسلمون بترجمة المخطوطات اليونانية والفارسية وحفظها.",
        "تعد علوم البلاغة والنحو من أهم ركائز دراسة النصوص الأدبية القديمة.",
        "تضم المكتبات التاريخية آلاف المخطوطات النادرة في الطب والفلك والجغرافيا.",
        "ساعدت تقنيات الرقمنة الحديثة على صيانة الوثائق التراثية وحمايتها من التلف.",
        "يحتوي الشعر العربي الجاهلي على تصوير دقيق للحياة الصحراوية والتقاليد الأصيلة.",
        "لعبت الأندلس دوراً محورياً في نقل العلوم والمعارف الشرقية إلى القارة الأوروبية.",
        "تاريخ الخطوط العربية يشمل الكوفي والنسخ والثلث والديواني والرقعة.",
        "إن دراسة وتحقيق المخطوطات التاريخية تتطلب معرفة عميقة باللغة وأصول التوثيق.",
        "ساهم ابن خلدون في تأسيس علم الاجتماع من خلال مقدمته التاريخية الشهيرة.",
        "القرآن الكريم هو المعجزة الخالدة التي حفظت اللغة العربية وقواعدها.",
        "تعتبر فهارس المخطوطات دليلاً للباحثين للوصول إلى كنوز التراث الأصيل.",
        "صناعة الورق وتجليد الكتب ازدهرت في المدن الإسلامية الكبرى كدمشق والقاهرة.",
        "العلم نور والجهل ظلام والمعرفة تبني المجتمعات وتفتح آفاق المستقبل المشرق.",
        "اللغة العربية لغة الضاد وهي من أثرى لغات العالم بالمفردات والمعاني البلاغية.",
        "القراءة غذاء العقل وبها ترتقي الأمم وتزدهر الثقافات والعلوم الإنسانية جمعاء.",
        "التاريخ يسجل إنجازات الحضارات الكبرى التي أثرت البشرية بالفلسفة والطب والفلك.",
        "الحفاظ على المخطوطات والآثار التاريخية واجب حضاري يحمي هوية الأجيال القادمة.",
    ],
    "mr": [
        "प्राचीन हस्तलिखिते महाराष्ट्राच्या समृद्ध ऐतिहासिक आणि सांस्कृतिक वारशाचे प्रतीक आहेत.",
        "ज्ञानेश्वरांनी भावार्थदीपिका लिहून मराठी भाषेला वैश्विक आध्यात्मिक अधिष्ठान दिले.",
        "संत तुकारामांचे अभंग आजही प्रत्येक मराठी मनाला प्रेरणा आणि मार्गदर्शन देतात.",
        "छत्रपती शिवाजी महाराजांच्या काळातील आज्ञापत्रे आणि पत्रव्यवहार ऐतिहासिक दृष्ट्या महत्त्वाचा आहे.",
        "मराठी भाषा आणि मोडी लिपीतील कागदपत्रे पुण्याच्या पेशवे दफ्तरात सुरक्षित आहेत.",
        "संत साहित्याने महाराष्ट्रात सामाजिक ऐक्य आणि समतेचा पाया रचला.",
        "आधुनिक मराठी साहित्यात कथा, कादंबरी आणि नाटकांची उज्ज्वल परंपरा आहे.",
        "पुरातन पोथ्यांचे संवर्धन करणे हे आपल्या संस्कृतीचे रक्षण करण्यासारखे आहे.",
        "भाषा संशोधकांनी मराठी भाषेच्या अभिजात दर्जाचे पुरावे संकलित केले आहेत.",
        "मराठी संस्कृतीमध्ये लोककला, लावणी आणि पोवाड्यांना विशेष महत्त्व आहे.",
        "इतिहास संशोधक वि. का. राजवाडे यांनी गावोगावी फिरून हस्तलिखिते गोळा केली.",
        "नाट्यसम्राट बालगंधर्वांनी मराठी संगीत रंगभूमीला सुवर्णकाळ मिळवून दिला.",
        "संत एकनाथांनी भागवताच्या माध्यमातून समाजाला सदाचाराचा मार्ग दाखवला.",
        "मराठी भाषा ही संस्कृत आणि प्राकृतपासून विकसित झालेली अतिशय समृद्ध भाषा आहे.",
        "प्राचीन हस्तलिखित ग्रंथांचे डिजिटायझेशन भावी पिढ्यांसाठी दिशादर्शक ठरेल.",
    ],
    "bn": [
        "প্রাচীন পুথিগুলি বাংলার সমৃদ্ধ সাহিত্য ও সাংস্কৃতিক ঐতিহ্যের অমূল্য নিদর্শন।",
        "চর্যাপদ বাংলা ভাষার প্রাচীনতম কাব্য ও গানের ঐতিহাসিক সংকলন হিসেবে স্বীকৃত।",
        "রবীন্দ্রনাথ ঠাকুর বাংলা সাহিত্যকে বিশ্বমঞ্চে এক অনন্য উচ্চতায় পৌঁছে দিয়েছেন।",
        "কাজী নজরুল ইসলামের বিদ্রোহী কবিতা স্বাধীনতার প্রেরণা জুগিয়েছিল।",
        "শ্রীচৈতন্যের বৈষ্ণব পদাবলী মধ্যযুগীয় বাংলা সাহিত্যকে আধ্যাত্মিক রসে সমৃদ্ধ করেছে।",
        "বাংলা লিপি ও ভাষার উৎপত্তি প্রাচীন মাগধী প্রাকৃত থেকে হয়েছে।",
        "বঙ্কিমচন্দ্র চট্টোপাধ্যায়ের উপন্যাসগুলি আধুনিক বাংলা গদ্যের ভিত্তি স্থাপন করে।",
        "কলকাতা বিশ্ববিদ্যালয়ের পুথিশালায় বহু প্রাচীন তালপাতার পুথি সংরক্ষিত আছে।",
        "লোকসাহিত্য ও বাউল গান গ্রামবাংলার চিরন্তন সুরকে ধারণ করে।",
        "ভাষাশহীদদের আত্মত্যাগ বাংলা ভাষাকে আন্তর্জাতিক মাতৃভাষা দিবসের মর্যাদা এনে দিয়েছে।",
        "মাইকেল মধুসূদন দত্ত অমিত্রাক্ষর ছন্দের প্রবর্তন করে বাংলা কাব্যকে নবজীবন দেন।",
        "প্রাচীন পাণ্ডুলিপি সংরক্ষণ ঐতিহ্য রক্ষা ও গবেষণার জন্য অত্যন্ত জরুরি।",
        "ঈশ্বরচন্দ্র বিদ্যাসাগর বাংলা বর্ণমালা ও গদ্যরীতির বিজ্ঞানসম্মত সংস্কার করেন।",
        "বাংলার সংস্কৃতিতে পূজা, পার্বণ ও সাহিত্য উৎসবের গভীর স্থান রয়েছে।",
        "ঐতিহাসিক নথিপত্র বিশ্লেষণ বাংলার সামাজিক পরিবর্তনের ধারাকে স্পষ্ট করে।",
    ],
    "fr": [
        "Les manuscrits anciens conservent la philosophie et la culture des siècles passés.",
        "La Bibliothèque nationale de France abrite des codex médiévaux d'une valeur inestimable.",
        "La littérature française a grandement influencé la pensée humaniste européenne.",
        "L'enluminure médiévale témoigne du savoir-faire artistique exceptionnel des copistes.",
        "La préservation numérique des archives historiques protège le patrimoine mondial.",
    ],
    "de": [
        "Alte Manuskripte bewahren das Wissen und die Philosophie vergangener Jahrhunderte.",
        "Die Entdeckung alter Schriften ermöglicht neue Einblicke in historische Ereignisse.",
        "Mittelalterliche Buchmalereien spiegeln die kulturellen Werte ihrer Epoche wider.",
        "Die Digitalisierung von Bibliotheken sichert wissenschaftliche Erkenntnisse für die Zukunft.",
        "Deutsche Gelehrte leisteten bedeutende Beiträge zur vergleichenden Sprachwissenschaft.",
        "Berlin ist die Bundeshauptstadt und Regierungssitz der Bundesrepublik Deutschland.",
        "Im Anfang war das Wort, und das Wort war bei Gott, und Gott war das Wort.",
        "Guten Morgen, wir hoffen, dass Sie einen wunderbaren und erfolgreichen Tag verbringen werden.",
        "Das historische Archiv enthält seltene Dokumente über die Entwicklung der deutschen Kultur.",
        "Wissenschaft und Forschung sind die tragenden Säulen des modernen gesellschaftlichen Fortschritts.",
        "Die Schönheit der Natur und die Ruhe der Wälder inspirieren Dichter seit Jahrhunderten.",
        "Kulturelles Erbe muss für zukünftige Generationen sorgfältig bewahrt und gepflegt werden.",
        "Johann Wolfgang von Goethe und Friedrich Schiller prägten die Epoche der Weimarer Klassik.",
        "Jeder Mensch hat das Recht auf freie Entfaltung seiner Persönlichkeit und Menschenwürde.",
        "In den Bibliotheken von Heidelberg und Göttingen lagern kostbare mittelalterliche Handschriften.",
        "Die deutsche Sprache zeichnet sich durch einen großen Reichtum an zusammengesetzten Wörtern aus.",
        "Philosophen wie Immanuel Kant untersuchten die Grundlagen der menschlichen Vernunft und Moral.",
        "Es war einmal ein König, der lebte in einem prächtigen Schloss am Rande eines großen Waldes.",
        "Die Reformation im sechzehnten Jahrhundert veränderte die europäische Geschichte grundlegend.",
        "Archäologische Funde belegen frühe Siedlungsspuren im Rheintal aus der Römerzeit.",
        "Bildung und Wissen sind die wertvollsten Güter, die eine Gesellschaft ihren Kindern mitgeben kann.",
        "Die Übersetzung historischer Manuskripte erfordert tiefes sprachliches und kulturelles Verständnis.",
        "Ein feste Burg ist unser Gott, ein gute Wehr und Waffen.",
        "Geduld und Ausdauer führen oft zum gewünschten Erfolg im Leben und im Beruf.",
        "Die Technologie von heute gestaltet die Welt und das Zusammenleben von morgen.",
    ],
    "la": [
        "In principio erat Verbum et Verbum erat apud Deum et Deus erat Verbum.",
        "Pater noster qui es in caelis sanctificetur nomen tuum adveniat regnum tuum.",
        "Cumque observarent senes diem aptum ingressa est Susanna sicut heri et nudiustertius.",
        "Nemo nos videt et fores pomerii clausae sunt et nos in concupiscentia tui sumus.",
        "Veritas vos liberabit et lux in tenebris lucet et tenebrae eam non comprehenderunt.",
        "Alea iacta est dixit Iulius Caesar cum Rubiconem flumen transiret.",
        "Veni vidi vici litteris mandavit post pugnam ponticam senatui Romano.",
        "Carpe diem quam minimum credula postero scripsit clarus poeta Horatius.",
        "Gloria in excelsis Deo et in terra pax hominibus bonae voluntatis.",
        "Requiem aeternam dona eis Domine et lux perpetua luceat eis in saecula saeculorum.",
        "Sanctus sanctus sanctus Dominus Deus Sabaoth pleni sunt caeli et terra gloria tua.",
        "Omnia vincit amor et nos cedamus amori cecinit Vergilius poeta Mantuanus.",
        "Cogito ergo sum principium fundamentale philosophiae modernae a Cartesio enuntiatum.",
        "Manuscripta antiqua monasteriorum memoriam saeculorum fideliter conservant.",
        "Deus caritas est et qui manet in caritate in Deo manet et Deus in eo.",
        "Audaces fortuna iuvat timidosque repellit proverbium antiquissimum est.",
        "Mens sana in corpore sano orandum est ut sit in omnibus hominibus.",
        "Tempus fugit inreparabile tempus dum loquimur aetas invida fugit.",
        "Historia magistra vitae et lux veritatis memoria antiquitatis nuntia vetustatis.",
        "Pax vobiscum dixit discipulis suis post resurrectionem dominus noster.",
    ],
    "es": [
        "Los manuscritos antiguos preservan el conocimiento y la filosofía de civilizaciones pasadas.",
        "El Archivo General de Indias contiene documentos históricos de gran valor cultural.",
        "La literatura clásica española refleja las tradiciones y el humanismo del Siglo de Oro.",
        "La conservación de pergaminos medievales requiere técnicas especializadas de restauración.",
        "El estudio paleográfico de los códices revela la evolución del idioma a lo largo del tiempo.",
    ],
}


def build_and_evaluate_model():
    """Build dataset, train/test split, evaluate accuracy, and save classifier."""
    texts = []
    labels = []

    for lang_code, samples in CORPUS_DATA.items():
        for text in samples:
            texts.append(text)
            labels.append(lang_code)

    print(f"Total samples collected across {len(CORPUS_DATA)} languages: {len(texts)}")

    # Split 80% train, 20% test stratified
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )

    print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # Character n-grams (1-3) Naive Bayes pipeline
    pipeline = Pipeline(
        [
            ("vectorizer", CountVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"Measured Test Accuracy: {accuracy * 100:.2f}% ({accuracy:.4f})")
    print("=" * 60)
    print("\nClassification Report on Held-Out Test Set:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save artifact
    output_dir = Path(__file__).resolve().parent.parent / "models" / "language_classifier"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "language_nb.pkl"

    payload = {
        "pipeline": pipeline,
        "classes": list(pipeline.classes_),
        "accuracy": float(accuracy),
        "ngram_range": (1, 3),
        "supported_languages": list(CORPUS_DATA.keys()),
    }

    joblib.dump(payload, model_path)
    print(f"\nModel successfully saved to: {model_path}")

    return accuracy


if __name__ == "__main__":
    build_and_evaluate_model()

