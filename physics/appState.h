
struct AppState
{
    AppState()
    {
        curFrame = FrameID(0, 0);
        gameModelPredictedCharacterFrame = FrameID(0, 0);
        curCharacterIndex = 1;
        anchorAnimationInstanceIndex = 0;
        showBBoxes = false;
        showFullMesh = false;
        showSelectionOnly = false;
        showTrackable = false;
        showCharacterSegments = false;
        showAnimationLabel = false;
        selectedSignature = (UINT64)-1;
        predictionIndex = 0;
    }

    UIConnection ui;
    EventMap eventMap;

    

    GeometryDatabase geoDatabase;
    SignatureColorMap colorMap;
    
    vector<D3D11TriMesh> curFrameMeshesBox;
    vector<D3D11TriMesh> curFrameMeshesFull;
    vector<D3D11TriMesh> curFrameMeshesRigidTransform;

    vector<D3D11TriMesh> gameModelFrameMeshesRigidTransform;

    SegmentAnalyzer analyzer;
    CharacterDatabase characters;
    ReplayDatabase replays;

    FrameID curFrame;
    FrameID anchorFrame;
    int curCharacterIndex;
    int anchorAnimationInstanceIndex;
    
    int predictionIndex;
    GameModel gameModel;
    GameState gameModelState;
    GameState gameModelStateStore;
    FrameID gameModelFrame;
    FrameID gameModelPredictedCharacterFrame;
    PredictionEntry gameModelPrediction;

    

    UINT64 selectedSignature;

    bool showBBoxes;
    bool showFullMesh;
    bool showSelectionOnly;
    bool showTrackable;
    bool showCharacterSegments;
    bool showAnimationLabel;
};
