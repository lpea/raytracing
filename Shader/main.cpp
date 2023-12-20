#include <Ogre.h>
#include <OgreApplicationContext.h>
#include <OgreCameraMan.h> // OgreBites::CameraMan
#include <OgreInput.h>
#include <OgreRTShaderSystem.h>

#include <iostream>

class ShaderTestApp : public OgreBites::ApplicationContext, public OgreBites::InputListener
{
public:
  ShaderTestApp();
  virtual ~ShaderTestApp() {}

  bool oneTimeConfig() override;
  void setup() override;
  Ogre::MaterialPtr loadMaterialFromFile();
  Ogre::MaterialPtr createMaterialByHand();
  bool frameStarted(const Ogre::FrameEvent& evt) override;
  bool frameEnded(const Ogre::FrameEvent& evt) override;
  bool keyPressed(const OgreBites::KeyboardEvent& evt) override;
};

ShaderTestApp::ShaderTestApp() : OgreBites::ApplicationContext("ShaderTestApp")
{
}

static const auto PREFERRED_RENDER_SYSTEM = "OpenGL 3+ Rendering Subsystem";

bool ShaderTestApp::oneTimeConfig()
{
  auto res = false;

  for (auto* renderer : getRoot()->getAvailableRenderers()) {
    if (renderer->getName() == PREFERRED_RENDER_SYSTEM) {
      res = true;
      renderer->setConfigOption("VSync", "No");
      renderer->setConfigOption("FSAA", "0");
      renderer->setConfigOption("Video Mode", "640 x 480");
      getRoot()->setRenderSystem(renderer);
      break;
    }
  }

  if (not res) {
    std::cout << "Unable to find preferred render subsystem. Using default policy..." << std::endl;
    res = OgreBites::ApplicationContext::oneTimeConfig();
  }

  return res;
}

void ShaderTestApp::setup()
{
  // do not forget to call the base first
  OgreBites::ApplicationContext::setup();
  addInputListener(this);

  // get a pointer to the already created root
  Ogre::Root* root = getRoot();
  Ogre::SceneManager* scene_manager = root->createSceneManager();

  // register our scene with the RTSS
  Ogre::RTShader::ShaderGenerator* shadergen = Ogre::RTShader::ShaderGenerator::getSingletonPtr();
  shadergen->addSceneManager(scene_manager);

  /////////////////////
  // Create a camera //
  /////////////////////

  Ogre::SceneNode* cam_node = scene_manager->getRootSceneNode()->createChildSceneNode();

  Ogre::Camera* cam = scene_manager->createCamera("cam");
  // cam->setNearClipDistance(5); // specific to this sample
  // cam->setAutoAspectRatio(true);
  // cam->setProjectionType(Ogre::ProjectionType::PT_PERSPECTIVE);
  // OgreBites::CameraMan* cam_man = new OgreBites::CameraMan(cam_node);
  // cam_man->setStyle(OgreBites::CameraStyle::CS_ORBIT);
  cam_node->attachObject(cam);
  cam_node->setPosition(0, 0, -1.5);
  // cam_node->setOrientation(Ogre::Quaternion(Ogre::Degree(15), Ogre::Vector3::UNIT_X));

  // Render into the main window
  auto vp = getRenderWindow()->addViewport(cam);
  vp->setAutoUpdated(true);

  ///////////////////////
  // Create a texture? //
  ///////////////////////

  // auto tex = Ogre::TextureManager::getSingleton().createManual(
  //     "texture",
  //     Ogre::RGN_DEFAULT,
  //     Ogre::TEX_TYPE_2D,
  //     640,
  //     480,
  //     0,
  //     Ogre::PF_A8B8G8R8,
  //     Ogre::TU_RENDERTARGET);

  ///////////////////////
  // Create a material //
  ///////////////////////

  // Method 1: load material from .material file
  auto mat = loadMaterialFromFile();

  // Method 2: create material programmatically
  // auto mat = createMaterialByHand();

  //////////////////////
  // Create an object //
  //////////////////////

  // Create Object (quad) -> (create Mesh?) -> attach Material (+ optional texture?)
  {
    auto* ogrObj = new Ogre::ManualObject("quad");
    ogrObj->begin(mat, Ogre::RenderOperation::OT_TRIANGLE_LIST);
    // set position, normal, color, textureCoord
    static const auto h = 1.0;
    ogrObj->position(Ogre::Vector3(-h, -h, 0));
    ogrObj->normal(Ogre::Vector3::UNIT_Z);
    ogrObj->position(Ogre::Vector3(h, -h, 0));
    ogrObj->normal(Ogre::Vector3::UNIT_Z);
    ogrObj->position(Ogre::Vector3(h, h, 0));
    ogrObj->normal(Ogre::Vector3::UNIT_Z);
    ogrObj->position(Ogre::Vector3(-h, h, 0));
    ogrObj->normal(Ogre::Vector3::UNIT_Z);
    // ogrObj->triangle(0, 1, 2);
    // ogrObj->triangle(0, 2, 3);
    ogrObj->quad(0, 1, 2, 3);
    ogrObj->end();
    scene_manager->getRootSceneNode()->attachObject(ogrObj);
  }
}

Ogre::MaterialPtr ShaderTestApp::loadMaterialFromFile()
{
  Ogre::MaterialPtr mat =
      Ogre::MaterialManager::getSingleton().getByName("TestMaterial", Ogre::RGN_DEFAULT);

  return mat;
}

Ogre::MaterialPtr ShaderTestApp::createMaterialByHand()
{
  Ogre::GpuProgramManager& mgr = Ogre::GpuProgramManager::getSingleton();

  // Load vertex program file
  // auto res_vert = Ogre::ResourceGroupManager::getSingleton().openResource(
  //     "vert.glsl", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
  Ogre::GpuProgramPtr vertex_program =
      mgr.createProgram("VertexShader", Ogre::RGN_DEFAULT, "glsl", Ogre::GPT_VERTEX_PROGRAM);
  // vertex_program->setSource(res_vert->getAsString());
  vertex_program->setSourceFile("vert.glsl");

  // Load fragment program file
  Ogre::GpuProgramPtr fragment_program =
      mgr.createProgram("FragmentShader", Ogre::RGN_DEFAULT, "glsl", Ogre::GPT_FRAGMENT_PROGRAM);
  fragment_program->setSourceFile("frag.glsl");

  // Create material
  Ogre::MaterialPtr mat =
      Ogre::MaterialManager::getSingleton().create("TestMaterialByHand", Ogre::RGN_DEFAULT);

  // Attach vertex and fragment programs to material
  auto* pass = mat->getTechnique(0)->getPass(0);
  pass->setVertexProgram("VertexShader");
  pass->setFragmentProgram("FragmentShader");

  // Set some parameters?
  // See OgreGpuProgramParams.h

  auto def_params = fragment_program->getDefaultParameters();
  def_params->setNamedAutoConstant("time", Ogre::GpuProgramParameters::ACT_TIME);
  def_params->setNamedAutoConstant("vp_width", Ogre::GpuProgramParameters::ACT_VIEWPORT_WIDTH);
  def_params->setNamedAutoConstant("vp_height", Ogre::GpuProgramParameters::ACT_VIEWPORT_HEIGHT);
  def_params->setNamedAutoConstant("camera_pos", Ogre::GpuProgramParameters::ACT_CAMERA_POSITION);

  return mat;
}

bool ShaderTestApp::frameStarted(const Ogre::FrameEvent& evt)
{
  const auto res = OgreBites::ApplicationContext::frameStarted(evt);
  std::cout << evt.timeSinceLastFrame << " s since last frame started" << std::endl;
  return res;
}

bool ShaderTestApp::frameEnded(const Ogre::FrameEvent& evt)
{
  const auto res = OgreBites::ApplicationContext::frameEnded(evt);
  std::cout << evt.timeSinceLastFrame << " s since last frame ended" << std::endl;
  return res;
}

bool ShaderTestApp::keyPressed(const OgreBites::KeyboardEvent& evt)
{
  if (evt.keysym.sym == OgreBites::SDLK_ESCAPE) {
    getRoot()->queueEndRendering();
  }
  return true;
}

int main(int /*argc*/, char** /*argv*/)
{
  try {
    ShaderTestApp app;
    app.initApp();
    app.getRoot()->startRendering();
    app.closeApp();
  } catch (const std::exception& e) {
    std::cerr << "Error occurred during execution: " << e.what() << '\n';
    return 1;
  }

  return 0;
}
