
#include "main.h"

void Vizzer::init(ApplicationData &app)
{
    assets.init(app.graphics);

    vec3f eye(1.0f, 1.0f, 4.5f);
    vec3f worldUp(0.0f, 1.0f, 0.0f);
    camera = Cameraf(-eye, eye, worldUp, 60.0f, (float)app.window.getWidth() / app.window.getHeight(), 0.01f, 10000.0f);

    font.init(app.graphics, "Calibri");

    physicsWorld.init();
}

void Vizzer::render(ApplicationData &app)
{
    timer.frame();

    const float borderRadius = 0.01f;
    assets.renderCylinder(camera.getCameraPerspective(), vec3f(1.0f, 0.0f, 0.0f), vec3f(1.0f, 1.0f, 0.0f), borderRadius, vec3f(1.0f, 0.0f, 0.0f));
    assets.renderCylinder(camera.getCameraPerspective(), vec3f(1.0f, 1.0f, 0.0f), vec3f(0.0f, 1.0f, 0.0f), borderRadius, vec3f(1.0f, 0.0f, 0.0f));
    assets.renderCylinder(camera.getCameraPerspective(), vec3f(0.0f, 1.0f, 0.0f), vec3f(0.0f, 0.0f, 0.0f), borderRadius, vec3f(1.0f, 0.0f, 0.0f));
    assets.renderCylinder(camera.getCameraPerspective(), vec3f(0.0f, 0.0f, 0.0f), vec3f(1.0f, 0.0f, 0.0f), borderRadius, vec3f(1.0f, 0.0f, 0.0f));

    vector<string> text;
    text.push_back("FPS: " + convert::toString(timer.framesPerSecond()));
    
    for (auto &body : physicsWorld.bodies)
    {
        assets.renderSphere(camera.getCameraPerspective(), body.position(), 1.0f, body.color);
        text.push_back("Pos: " + body.position().toString());
    }

    const bool useText = true;
    if (useText)
        drawText(app, text);
}

void Vizzer::resize(ApplicationData &app)
{
    camera.updateAspectRatio((float)app.window.getWidth() / app.window.getHeight());
}

void Vizzer::drawText(ApplicationData &app, vector<string> &text)
{
    int y = 0;
    for (auto &entry : text)
    {
        font.drawString(app.graphics, entry, vec2i(10, 5 + y++ * 25), 24.0f, RGBColor::Red);
    }
}

void Vizzer::keyDown(ApplicationData &app, UINT key)
{
    if (key == KEY_F) app.graphics.castD3D11().toggleWireframe();

    if (key == KEY_P)
    {
        PhysicsNetDatabase database;
        database.init();
        //database.createDatabase(R"(G:\PhysicsNet\databases\simpleDropPhysState2mil\)", 2000000);
        
        database.createDatabaseFlat(R"(G:\PhysicsNet\databases\simpleBall2k.txt)", 2000);
        //database.createDatabaseFlat(R"(G:\PhysicsNet\databases\simpleBall2mil.txt)", 2000000);
    }
}

void Vizzer::keyPressed(ApplicationData &app, UINT key)
{
    const float distance = 1.0f;
    const float theta = 5.0f;

    if (key == KEY_Z) physicsWorld.macroStep();
    if (key == KEY_R) physicsWorld.init();

    if (key == KEY_M)
    {
        ColorImageR8G8B8A8 image(128, 128);
        physicsWorld.render(PhysicsRender::random(), image);
        LodePNG::save(image, "debug.png");
    }

    if(key == KEY_S) camera.move(-distance);
    if(key == KEY_W) camera.move(distance);
    if(key == KEY_A) camera.strafe(-distance);
    if(key == KEY_D) camera.strafe(distance);
	if(key == KEY_E) camera.jump(distance);
	if(key == KEY_Q) camera.jump(-distance);

    if(key == KEY_UP) camera.lookUp(theta);
    if(key == KEY_DOWN) camera.lookUp(-theta);
    if(key == KEY_LEFT) camera.lookRight(theta);
    if(key == KEY_RIGHT) camera.lookRight(-theta);
}

void Vizzer::mouseDown(ApplicationData &app, MouseButtonType button)
{

}

void Vizzer::mouseWheel(ApplicationData &app, int wheelDelta)
{
    const float distance = 0.01f;
    camera.move(distance * wheelDelta);
}

void Vizzer::mouseMove(ApplicationData &app)
{
    const float distance = 0.1f;
    const float theta = 0.4f;

    vec2i posDelta = app.input.mouse.pos - app.input.prevMouse.pos;

    if(app.input.mouse.buttons[MouseButtonRight])
    {
        camera.strafe(-distance * posDelta.x);
        camera.jump(distance * posDelta.y);
    }

    if(app.input.mouse.buttons[MouseButtonLeft])
    {
        camera.lookRight(-theta * posDelta.x);
        camera.lookUp(theta * posDelta.y);
    }

}
