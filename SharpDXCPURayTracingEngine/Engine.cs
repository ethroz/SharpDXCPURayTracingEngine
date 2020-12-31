using SharpDX;
using SharpDX.Direct3D;
using SharpDX.Windows;
using SharpDX.DirectInput;
using SharpDX.DXGI;
using D3D11 = SharpDX.Direct3D11;
using System;
using System.Collections.Generic;
using System.Drawing;
using Color = System.Drawing.Color;
using SharpDX.Direct3D11;
using SharpDX.D3DCompiler;
using System.Runtime.InteropServices;
using System.Drawing.Imaging;
using System.IO;
using System.Threading.Tasks;
using System.Reflection;

namespace SharpDXCPURayTracingEngine
{
    sealed class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            using (Engine engine = new Engine())
            {
                engine.Run();
            }
        }
    }

    public class Engine : IDisposable
    {
        // Graphics Fields
        private RenderForm renderForm;
        private D3D11.Device device;
        private DeviceContext deviceContext;
        private SwapChain swapChain;
        private RenderTargetView renderTargetView;
        //private Vector4[] vertices = new Vector4[]
        //{
        //    new Vector4(-1.0f, -1.0f, 0.0f, 1.0f),
        //    new Vector4(-1.0f, 1.0f, 0.0f, 1.0f),
        //    new Vector4(1.0f, 1.0f, 0.0f, 1.0f),
        //    new Vector4(-1.0f, -1.0f, 0.0f, 1.0f),
        //    new Vector4(1.0f, 1.0f, 0.0f, 1.0f),
        //    new Vector4(1.0f, -1.0f, 0.0f, 1.0f)
        //};
        private VertexPositionTexture[] vertices = new VertexPositionTexture[]
        {
            new VertexPositionTexture(new Vector3(-1.0f, -1.0f, 0.0f), new Vector2(0.0f, 0.0f)),
            new VertexPositionTexture(new Vector3(-1.0f, 1.0f, 0.0f), new Vector2(0.0f, 1.0f)),
            new VertexPositionTexture(new Vector3(1.0f, -1.0f, 0.0f), new Vector2(1.0f, 0.0f)),
            new VertexPositionTexture(new Vector3(1.0f, 1.0f, 0.0f), new Vector2(1.0f, 1.0f))
        };
        private D3D11.Buffer triangleVertexBuffer;
        private VertexShader vertexShader;
        private PixelShader pixelShader;
        //private InputElement[] inputElements = new InputElement[]
        //{
        //    new InputElement("POSITION", 0, Format.R32G32B32_Float, 0, 0, InputClassification.PerVertexData, 0)
        //};
        private InputElement[] inputElements = new InputElement[]
        {
            new InputElement("POSITION", 0, Format.R32G32B32A32_Float, 0, 0, InputClassification.PerVertexData, 0),
            new InputElement("TEXCOORD", 0, Format.R32G32B32A32_Float, 16, 0, InputClassification.PerVertexData, 0)
        };
        private ShaderSignature inputSignature;
        private InputLayout inputLayout;
        private Viewport viewport;
        private Texture2DDescription td;
        private DirectBitmap dbmp;
        private Texture2D texture;
        private string path;

        // Input Fields
        private Keyboard keyboard;
        private Chey[] cheyArray;

        // Controllable Graphics Settings
        private double RefreshRate = 144;
        private int Width = 480, Height = 270; // not reccomended to exceed display size
        private bool Running = true;
        public enum WindowState { Normal, Minimized, Maximized, FullScreen };
        private WindowState State = WindowState.Normal;

        // Frame Fields
        public double elapsedTime;
        private long t1, t2;
        private System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();

        // Game Fields
        //private int RayCount = 1; // Will be sqaured
        private int RayDepth = 4;
        private List<Gameobject> gameobjects = new List<Gameobject>();
        private Light light;

        private bool Test()
        {
            //OnAwake();
            //var data = dbmp.Bitmap.LockBits(new System.Drawing.Rectangle(0, 0, Width, Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            //Console.WriteLine(data.Stride);
            return false;
        }

        public Engine()
        {
            OnAwake();
            if (Test())
            {
                Console.ReadKey();
                Environment.Exit(0);
            }

            renderForm = new RenderForm("SharpDXRayTracingEngine")
            {
                ClientSize = new Size(Width, Height),
                AllowUserResizing = true
            };
            if (State == WindowState.FullScreen)
            {
                renderForm.TopMost = true;
                renderForm.FormBorderStyle = System.Windows.Forms.FormBorderStyle.None;
                renderForm.WindowState = System.Windows.Forms.FormWindowState.Maximized;
            }
            else if (State == WindowState.Maximized)
            {
                renderForm.WindowState = System.Windows.Forms.FormWindowState.Maximized;
            }
            else if (State == WindowState.Minimized)
            {
                renderForm.TopMost = false;
                renderForm.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Sizable;
                renderForm.WindowState = System.Windows.Forms.FormWindowState.Minimized;
            }

            InitializeKeyboard();
            InitializeDeviceResources();
            InitializePlane();
            InitializeShaders();
            OnStart();
            
            t1 = sw.ElapsedTicks;
        }

        public void Run()
        {
            RenderLoop.Run(renderForm, RenderCallBack);
        }

        private void RenderCallBack()
        {
            GetTime();
            GetKeys();
            UserInput();
            if (!Running)
                return;
            OnUpdate();
            Draw();
        }

        private void InitializeKeyboard()
        {
            keyboard = new Keyboard(new DirectInput());
            keyboard.Properties.BufferSize = 128;
            keyboard.Acquire();
            var state = keyboard.GetCurrentState();
            var allKeys = state.AllKeys;
            cheyArray = new Chey[allKeys.Count];
            for (int i = 0; i < allKeys.Count; i++)
                cheyArray[i] = new Chey(allKeys[i]);
        }

        private void InitializeDeviceResources()
        {
            SwapChainDescription swapChainDesc = new SwapChainDescription()
            {
                ModeDescription = new ModeDescription(Width, Height, new Rational(10000, 1), Format.R8G8B8A8_UNorm),
                SampleDescription = new SampleDescription(1, 0),
                Usage = Usage.RenderTargetOutput,
                BufferCount = 1,
                OutputHandle = renderForm.Handle,
                IsWindowed = true,
            };
            D3D11.Device.CreateWithSwapChain(DriverType.Hardware, DeviceCreationFlags.None, swapChainDesc, out device, out swapChain);
            deviceContext = device.ImmediateContext;
            using (Texture2D backBuffer = swapChain.GetBackBuffer<Texture2D>(0))
            {
                renderTargetView = new RenderTargetView(device, backBuffer);
            }
            viewport = new Viewport(0, 0, Width, Height);
            deviceContext.Rasterizer.SetViewport(viewport);
        }

        private void InitializePlane()
        {
            triangleVertexBuffer = D3D11.Buffer.Create(device, BindFlags.VertexBuffer, vertices);
        }

        private void InitializeShaders()
        {
            using (var vertexShaderByteCode = ShaderBytecode.CompileFromFile("Shaders.hlsl", "vertexShader", "vs_5_0", ShaderFlags.Debug))
            {
                inputSignature = ShaderSignature.GetInputSignature(vertexShaderByteCode);
                vertexShader = new VertexShader(device, vertexShaderByteCode);
            }
            using (var pixelShaderByteCode = ShaderBytecode.CompileFromFile("Shaders.hlsl", "pixelShader", "ps_5_0", ShaderFlags.Debug))
            {
                pixelShader = new PixelShader(device, pixelShaderByteCode);
            }
            deviceContext.VertexShader.Set(vertexShader);
            deviceContext.PixelShader.Set(pixelShader);

            var samplerStateDescription = new SamplerStateDescription
            {
                AddressU = TextureAddressMode.Wrap,
                AddressV = TextureAddressMode.Wrap,
                AddressW = TextureAddressMode.Wrap,
                Filter = Filter.MinMagMipLinear
            };
            using (var samplerState = new SamplerState(device, samplerStateDescription))
                deviceContext.PixelShader.SetSampler(0, samplerState);

            inputLayout = new InputLayout(device, inputSignature, inputElements);
            deviceContext.InputAssembler.InputLayout = inputLayout;

            deviceContext.InputAssembler.PrimitiveTopology = PrimitiveTopology.TriangleStrip;
        }

        private void GetTime()
        {
            t2 = sw.ElapsedTicks;
            elapsedTime = (t2 - t1) / 10000000.0;
            if (RefreshRate != 0)
            {
                while (1.0 / elapsedTime > RefreshRate)
                {
                    t2 = sw.ElapsedTicks;
                    elapsedTime = (t2 - t1) / 10000000.0;
                }
            }
            t1 = t2;
            renderForm.Text = "SharpDXCPURayTracingEngine   FPS:" + (1.0 / (elapsedTime)).ToString("G4");
        }

        private void GetKeys()
        {
            keyboard.Poll();
            var state = keyboard.GetCurrentState();
            for (int i = 0; i < cheyArray.Length; i++)
            {
                bool pressed = state.IsPressed(cheyArray[i].key);
                cheyArray[i].Down = cheyArray[i].Raised && pressed;
                cheyArray[i].Up = cheyArray[i].Held && !pressed;
                cheyArray[i].Held = pressed;
                cheyArray[i].Raised = !pressed;
            }
        }

        public bool KeyDown(Key key)
        {
            return FindChey(key).Down;
        }

        public bool KeyUp(Key key)
        {
            return FindChey(key).Up;
        }

        public bool KeyHeld(Key key)
        {
            return FindChey(key).Held;
        }

        public bool KeyRaised(Key key)
        {
            return FindChey(key).Raised;
        }

        private Chey FindChey(Key key)
        {
            for (int i = 0; i < cheyArray.Length; i++)
            {
                if (cheyArray[i].key == key)
                    return cheyArray[i];
            }
            return null;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool boolean)
        {
            device.Dispose();
            deviceContext.Dispose();
            swapChain.Dispose();
            renderTargetView.Dispose();
            triangleVertexBuffer.Dispose();
            vertexShader.Dispose();
            pixelShader.Dispose();
            inputLayout.Dispose();
            inputSignature.Dispose();
            renderForm.Dispose();
        }

        /////////////////////////////////////

        private void Draw()
        {
            texture = new Texture2D(device, td, new DataRectangle(dbmp.BitsHandle.AddrOfPinnedObject(), Width * 4));
            ShaderResourceView textureView = new ShaderResourceView(device, texture);
            deviceContext.PixelShader.SetShaderResource(0, textureView);
            texture.Dispose();
            textureView.Dispose();

            deviceContext.OutputMerger.SetRenderTargets(renderTargetView);
            deviceContext.InputAssembler.SetVertexBuffers(0, new VertexBufferBinding(triangleVertexBuffer, Utilities.SizeOf<VertexPositionTexture>(), 0));
            deviceContext.Draw(vertices.Length, 0);

            swapChain.Present(0, PresentFlags.None);
        }

        public void OnAwake()
        {
            sw.Start();
            path = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location);
            dbmp = new DirectBitmap(Width, Height);
            td = new Texture2DDescription
            {
                Width = dbmp.Width,
                Height = dbmp.Height,
                ArraySize = 1,
                BindFlags = BindFlags.ShaderResource,
                Usage = ResourceUsage.Immutable,
                CpuAccessFlags = CpuAccessFlags.None,
                Format = Format.B8G8R8A8_UNorm,
                MipLevels = 1,
                OptionFlags = ResourceOptionFlags.None,
                SampleDescription = new SampleDescription(1, 0),
            };
        }

        public void OnStart()
        {
            //renderForm.Size = new Size(480, 270);
            //gameobjects.AddRange(GetObjectsFromFile(@"C:\Users\ethro\source\repos\SharpDXCPURayTracingEngine\SharpDXCPURayTracingEngine\Object\Objects.obj"));
            //for (int i = 0; i < gameobjects.Count; i++)
            //{
            //    gameobjects[i].position += new Vec3D(0.0f, -2.0f, 5.0f);
            //}
            gameobjects.Add(GetObjectFromFile(path + @"\Object\Cube.obj"));
            gameobjects[0].position = new Vec3D(0.0f, 0.0f, 3.0f);
            light = new Light(new Vec3D(-1.0f, 0.0f, 3.0f), 0.1f, 1000.0f);
            Tria[] planeTris = new Tria[]
            {
                new Tria(new Vec3D(-100.0f, 0.0f, -100.0f), new Vec3D(-100.0f, 0.0f, 100.0f), new Vec3D(100.0f, 0.0f, 100.0f), Color.Blue),
                new Tria(new Vec3D(-100.0f, 0.0f, -100.0f), new Vec3D(100.0f, 0.0f, 100.0f), new Vec3D(100.0f, 0.0f, -100.0f), Color.Blue)
            };
            gameobjects.Add(new Gameobject(new Vec3D(0.0f, -2.0f, 0.0f), new Vec3D(), new Vec3D(1.0f), planeTris, new Material(0.5f, 1.0f)));
        }

        public void OnUpdate()
        {
            // vertex shader
            for (int g = 0; g < gameobjects.Count; g++)
            {
                gameobjects[g].projectedTriangles = new Tria[gameobjects[g].triangles.Length];
                for (int t = 0; t < gameobjects[g].triangles.Length; t++)
                {
                    Tria temp = new Tria(gameobjects[g].triangles[t].points[0], gameobjects[g].triangles[t].points[1], gameobjects[g].triangles[t].points[2], gameobjects[g].triangles[t].color);
                    for (int i = 0; i < 3; i++)
                    {
                        temp.points[i] += gameobjects[g].position;
                    }
                    gameobjects[g].projectedTriangles[t] = temp;
                }
            }

            //pixel shader
            Parallel.For(0, Height, y =>
            {
                float pitch = (y * 2.0f / Height - 1.0f) * (Height / (float)Width) * 0.1f;
                for (int x = 0; x < Width; x++)
                {
                    float yaw = (x * 2.0f / Width - 1.0f) * 0.1f;
                    Vec3D direction = new Vec3D(yaw, pitch, 0.1f);
                    Ray ray = new Ray(direction, direction * 2);
                    Color color;
                    List<Color> colors = new List<Color>();
                    float[] impact = new float[RayDepth];
                    float[] distances = new float[RayDepth];
                    for (int i = 0; i < RayDepth; i++)
                    {
                        colors.Add(RayCast(ref ray, i, ref distances, out impact[i]));
                        if (distances[i] == float.PositiveInfinity)
                            break;
                    }
                    int r = 0, g = 0, b = 0;
                    for (int i = 0; i < colors.Count; i++)
                    {
                        r += (int)(colors[i].R * impact[i]);
                        g += (int)(colors[i].G * impact[i]);
                        b += (int)(colors[i].B * impact[i]);
                    }
                    r = Math.Min(255, r / colors.Count);
                    g = Math.Min(255, g / colors.Count);
                    b = Math.Min(255, b / colors.Count);
                    color = Color.FromArgb(r, g, b);
                    dbmp.SetPixel(x, y, color);
                }
            });
        }

        private Color RayCast(ref Ray ray, int iteration, ref float[] distances, out float impact)
        {
            float bestDistance = float.PositiveInfinity;
            int[] indices = new int[2];
            for (int i = 0; i < gameobjects.Count; i++)
            {
                for (int j = 0; j < gameobjects[i].triangles.Length; j++)
                {
                    float distance = TriangleIntersect(ray, gameobjects[i].projectedTriangles[j]);
                    if (distance != -1.0f && distance < bestDistance)   // check that the triangle intersects ray and is the closest 
                    {
                        bestDistance = distance;
                        indices[0] = i;
                        indices[1] = j;
                    }
                }
            }
            float lightDistance = LightIntersect(ray, light);
            if (lightDistance != -1.0f && lightDistance < bestDistance) // check if the light is closer
            {
                distances[iteration] += lightDistance;
                impact = light.luminosity;
                return Color.White;
            }
            distances[iteration] += bestDistance;
            if (bestDistance == float.PositiveInfinity) // check if the ray missed
            {
                impact = 0.0f;
                return Color.Black;
            }
            ray.origin = ray.origin + ray.direction * bestDistance;
            ray.Reflect(gameobjects[indices[0]].projectedTriangles[indices[1]].normal); // reflect the ray off of the surface for the next raycast
            Color color = gameobjects[indices[0]].triangles[indices[1]].color;
            Vec3D toLight = light.position - ray.origin;
            if (Vec3D.Dot(toLight, gameobjects[indices[0]].projectedTriangles[indices[1]].normal) <= 0) // check if the surface is opposing the light
            {
                impact = 0.0f;
                return Color.Black;
            }
            float distanceFromLight = toLight.Length();
            //for (int j = 0; j < iteration; j++)
            //{
            //    distanceFromLight += distances[j];  // cumulative distance to eye
            //}
            float brightness = light.luminosity / 4 / (float)Math.PI / Square(distanceFromLight);
            if (brightness > 1)
            {
                impact = brightness;
                return ColorLerp(color, Color.White, (brightness - 1) / 100);
            }
            else
            {
                impact = 1.0f;
                return ColorLerp(Color.Black, color, brightness);
            }
        }

        private float TriangleIntersect(Ray ray, Tria tri)
        {
            float numerator = Vec3D.Dot(tri.normal, tri.points[0] - ray.origin);
            float denominator = Vec3D.Dot(tri.normal, ray.direction);
            if (denominator >= 0.0f)
            {
                return -1.0f;
            }
            float intersection = numerator / denominator;
            if (intersection < 0.0f)
            {
                return -1.0f;
            }

            // test if intersection is inside triangle ////////////////////////////

            Vec3D point = ray.origin + ray.direction * intersection;
            Vec3D edge0 = tri.points[1] - tri.points[0];
            Vec3D edge1 = tri.points[2] - tri.points[1];
            Vec3D edge2 = tri.points[0] - tri.points[2];
            Vec3D C0 = point - tri.points[0];
            Vec3D C1 = point - tri.points[1];
            Vec3D C2 = point - tri.points[2];
            if (!(Vec3D.Dot(tri.normal, Vec3D.Cross(C0, edge0)) >= 0 &&
                Vec3D.Dot(tri.normal, Vec3D.Cross(C1, edge1)) >= 0 &&
                Vec3D.Dot(tri.normal, Vec3D.Cross(C2, edge2)) >= 0))
            {
                return -1.0f; // point is inside the triangle
            }
            return intersection;
        }

        private float LightIntersect(Ray ray, Light light)
        {
            Vec3D toSphere = ray.origin - light.position;
            float discriminant = Square(Vec3D.Dot(ray.direction, toSphere)) - toSphere.LengthSquared() + Square(light.radius);
            if (discriminant <= 0.0f)
            {
                return -1.0f;
            }
            float intersection = -Vec3D.Dot(ray.direction, ray.origin - light.position) - (float)Math.Sqrt(discriminant);
            if (intersection < 0.0f)
            {
                return -1.0f;
            }
            return intersection;
        }

        private Color ColorLerp(Color x, Color y, float t)
        {
            if (t >= 1)
                return y;
            else if (t <= 0)
                return x;
            int r = x.R + (int)((y.R - x.R) * t),
                g = x.G + (int)((y.G - x.G) * t),
                b = x.B + (int)((y.B - x.B) * t);
            return Color.FromArgb(r, g, b);
        }

        private Gameobject GetObjectFromFile(string FileName)
        {
            if (!File.Exists(FileName))
                return null;
            string[] document = File.ReadAllLines(FileName);
            if (document == null)
                return null;

            Tria[] triangles;
            List<Vec3D> verts = new List<Vec3D>();
            List<Color> colors = new List<Color>();
            List<int[]> indices = new List<int[]>();

            for (int i = 0; i < document.Length; i++)
            {

                if (document[i] == "" || document[i][0] == '#' || document[i][0] == 'o')
                    continue;
                else if (document[i][0] == 'v')
                {
                    List<int> values = new List<int>();
                    for (int j = 1; j < document[i].Length; j++)
                    {
                        if (document[i][j] == ' ')
                            values.Add(j + 1);
                    }

                    Vec3D v = new Vec3D
                    (
                        float.Parse(document[i].Substring(values[0], values[1] - values[0] - 1)),
                        float.Parse(document[i].Substring(values[1], values[2] - values[1] - 1)),
                        float.Parse(document[i].Substring(values[2], values[3] - values[2] - 1))
                    );

                    Color c = Color.FromArgb
                    (
                        int.Parse(document[i].Substring(values[3], values[4] - values[3] - 1)),
                        int.Parse(document[i].Substring(values[4], values[5] - values[4] - 1)),
                        int.Parse(document[i].Substring(values[5]))
                    );

                    colors.Add(c);
                    verts.Add(v);
                }
                else if (document[i][0] == 'f')
                {
                    List<int> values = new List<int>();
                    for (int j = 1; j < document[i].Length; j++)
                    {
                        if (document[i][j] == ' ')
                            values.Add(j + 1);
                    }
                    int[] lineNum = new int[3]
                    {
                        int.Parse(document[i].Substring(values[0], values[1] - values[0] - 1)),
                        int.Parse(document[i].Substring(values[1], values[2] - values[1] - 1)),
                        int.Parse(document[i].Substring(values[2]))
                    };

                    indices.Add(new int[] { lineNum[0] - 1, lineNum[1] - 1, lineNum[2] - 1 });
                }
            }

            Vec3D position = new Vec3D();
            for (int i = 0; i < verts.Count; i++)
                position += verts[i];
            position /= verts.Count;
            for (int i = 0; i < verts.Count; i++)
                verts[i] -= position;
            triangles = new Tria[indices.Count];
            for (int i = 0; i < indices.Count; i++)
            {
                int R = (colors[indices[i][0]].R + colors[indices[i][1]].R + colors[indices[i][2]].R) / 3;
                int G = (colors[indices[i][0]].G + colors[indices[i][1]].G + colors[indices[i][2]].G) / 3;
                int B = (colors[indices[i][0]].B + colors[indices[i][1]].B + colors[indices[i][2]].B) / 3;
                Color averageColor = Color.FromArgb(R, G, B);
                triangles[i] = new Tria(verts[indices[i][0]], verts[indices[i][1]], verts[indices[i][2]], averageColor);
            }

            return new Gameobject(new Vec3D(), new Vec3D(), new Vec3D(1.0f, 1.0f, 1.0f), triangles, new Material(0.5f, 1.0f));
        }

        private Gameobject[] GetObjectsFromFile(string FileName)
        {
            if (!File.Exists(FileName))
                return null;
            string[] document = File.ReadAllLines(FileName);
            if (document == null)
                return null;

            List<Gameobject> objs = new List<Gameobject>();
            List<Vec3D> verts = new List<Vec3D>();
            List<Color> colors = new List<Color>();
            Tria[] triangles;
            List<int[]> indices = new List<int[]>();
            int verticesCount = 0;
            int vertexCount = 0;
            int objectIndex = -1;

            for (int i = 0; i < document.Length; i++)
            {

                if (document[i] == "" || document[i][0] == '#' || document[i][0] == 'm')
                    continue;
                else if (document[i][0] == 'o')
                {
                    if (objectIndex > -1)
                    {
                        triangles = new Tria[indices.Count];
                        Vec3D pos = new Vec3D();
                        for (int j = 0; j < verts.Count; j++)
                            pos += verts[j];
                        pos /= verts.Count;
                        for (int j = 0; j < verts.Count; j++)
                            verts[j] -= pos;
                        for (int j = 0; j < indices.Count; j++)
                        {
                            int R = (colors[indices[j][0]].R + colors[indices[j][1]].R + colors[indices[j][2]].R) / 3;
                            int G = (colors[indices[j][0]].G + colors[indices[j][1]].G + colors[indices[j][2]].G) / 3;
                            int B = (colors[indices[j][0]].B + colors[indices[j][1]].B + colors[indices[j][2]].B) / 3;
                            Color averageColor = Color.FromArgb(R, G, B);
                            triangles[j] = new Tria(verts[indices[j][0]], verts[indices[j][1]], verts[indices[j][2]], averageColor);
                        }
                        objs[objectIndex].position = pos;
                        objs[objectIndex].triangles = triangles;
                        vertexCount += verticesCount;
                        verticesCount = 0;
                    }
                    objs.Add(new Gameobject());
                    verts = new List<Vec3D>();
                    colors = new List<Color>();
                    indices = new List<int[]>();
                    objectIndex++;
                }
                else if (document[i][0] == 'v')
                {
                    List<int> values = new List<int>();
                    for (int j = 1; j < document[i].Length; j++)
                    {
                        if (document[i][j] == ' ')
                            values.Add(j + 1);
                    }

                    Vec3D v = new Vec3D
                    (
                        float.Parse(document[i].Substring(values[0], values[1] - values[0] - 1)),
                        float.Parse(document[i].Substring(values[1], values[2] - values[1] - 1)),
                        float.Parse(document[i].Substring(values[2], values[3] - values[2] - 1))
                    );

                    Color c = Color.FromArgb
                    (
                        int.Parse(document[i].Substring(values[3], values[4] - values[3] - 1)),
                        int.Parse(document[i].Substring(values[4], values[5] - values[4] - 1)),
                        int.Parse(document[i].Substring(values[5]))
                    );

                    verts.Add(v);
                    colors.Add(c);
                    verticesCount++;
                }
                else if (document[i][0] == 'f')
                {
                    List<int> values = new List<int>();
                    for (int j = 1; j < document[i].Length; j++)
                    {
                        if (document[i][j] == ' ')
                            values.Add(j + 1);
                    }
                    int[] lineNum = new int[3]
                    {
                        int.Parse(document[i].Substring(values[0], values[1] - values[0] - 1)),
                        int.Parse(document[i].Substring(values[1], values[2] - values[1] - 1)),
                        int.Parse(document[i].Substring(values[2]))
                    };

                    indices.Add(new int[] { lineNum[0] - 1 - vertexCount, lineNum[1] - 1 - vertexCount, lineNum[2] - 1 - vertexCount });
                }
            }
            triangles = new Tria[indices.Count];
            Vec3D position = new Vec3D();
            for (int i = 0; i < verts.Count; i++)
                position += verts[i];
            position /= verts.Count;
            for (int i = 0; i < verts.Count; i++)
                verts[i] -= position;
            for (int i = 0; i < indices.Count; i++)
            {
                int R = (colors[indices[i][0]].R + colors[indices[i][1]].R + colors[indices[i][2]].R) / 3;
                int G = (colors[indices[i][0]].G + colors[indices[i][1]].G + colors[indices[i][2]].G) / 3;
                int B = (colors[indices[i][0]].B + colors[indices[i][1]].B + colors[indices[i][2]].B) / 3;
                Color averageColor = Color.FromArgb(R, G, B);
                triangles[i] = new Tria(verts[indices[i][0]], verts[indices[i][1]], verts[indices[i][2]], averageColor);
            }
            objs[objectIndex].position = position;
            objs[objectIndex].triangles = triangles;

            return objs.ToArray();
        }

        public void UserInput()
        {
            if (KeyDown(Key.P))
                Running = !Running;
            if (!Running)
                return;

            if (KeyHeld(Key.W))
                gameobjects[0].position.z += (float)(1.6 * elapsedTime);
            if (KeyHeld(Key.S))
                gameobjects[0].position.z -= (float)(1.6 * elapsedTime);
            if (KeyHeld(Key.A))
                gameobjects[0].position.x -= (float)(1.6 * elapsedTime);
            if (KeyHeld(Key.D))
                gameobjects[0].position.x += (float)(1.6 * elapsedTime);
            if (KeyHeld(Key.Q))
                gameobjects[0].position.y -= (float)(1.6 * elapsedTime);
            if (KeyHeld(Key.E))
                gameobjects[0].position.y += (float)(1.6 * elapsedTime);

            if (KeyHeld(Key.I))
                light.position.z += (float)(1.6 * elapsedTime);
            if (KeyHeld(Key.K))
                light.position.z -= (float)(1.6 * elapsedTime);
            if (KeyHeld(Key.J))
                light.position.x -= (float)(1.6 * elapsedTime);
            if (KeyHeld(Key.L))
                light.position.x += (float)(1.6 * elapsedTime);

            if (KeyDown(Key.F11))
            {
                if (State != WindowState.Maximized)
                {
                    State = WindowState.Maximized;
                    renderForm.TopMost = false;
                    renderForm.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Sizable;
                    renderForm.WindowState = System.Windows.Forms.FormWindowState.Maximized;
                }
                else
                {
                    State = WindowState.Normal;
                    renderForm.TopMost = false;
                    renderForm.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Sizable;
                    renderForm.WindowState = System.Windows.Forms.FormWindowState.Normal;
                }
            }
            if (KeyDown(Key.Tab))
                CycleWindowState();
            if (KeyDown(Key.Escape))
                Environment.Exit(0);
        }

        /////////////////////////////////////

        public void CycleWindowState()
        {
            switch (State)
            {
                case WindowState.Minimized:
                    State = WindowState.Normal;
                    renderForm.TopMost = false;
                    renderForm.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Sizable;
                    renderForm.WindowState = System.Windows.Forms.FormWindowState.Normal;
                    break;
                case WindowState.Normal:
                    State = WindowState.Maximized;
                    renderForm.TopMost = false;
                    renderForm.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Sizable;
                    renderForm.WindowState = System.Windows.Forms.FormWindowState.Maximized;
                    break;
                case WindowState.Maximized:
                    State = WindowState.FullScreen;
                    renderForm.TopMost = true;
                    renderForm.FormBorderStyle = System.Windows.Forms.FormBorderStyle.None;
                    renderForm.WindowState = System.Windows.Forms.FormWindowState.Normal;
                    renderForm.WindowState = System.Windows.Forms.FormWindowState.Maximized;
                    break;
                case WindowState.FullScreen:
                    State = WindowState.Minimized;
                    renderForm.TopMost = false;
                    renderForm.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Sizable;
                    renderForm.WindowState = System.Windows.Forms.FormWindowState.Minimized;
                    break;
            }
        }

        public static float Square(float value)
        {
            return value * value;
        }

        ////////////////////////////////////

        private class Material
        {
            public float diffuse, specular;

            public Material(float d, float s)
            {
                diffuse = d;
                specular = s;
            }
        }

        private class Light
        {
            public Vec3D position;
            public float radius, luminosity;

            public Light(Vec3D p, float r, float l)
            {
                position = p;
                radius = r;
                luminosity = l;
            }
        }

        private class Gameobject
        {
            public Vec3D position, rotation, scale;
            public Material material;
            public Tria[] triangles;
            public Tria[] projectedTriangles;

            public Gameobject()
            {
                position = new Vec3D();
                rotation = new Vec3D();
                scale = new Vec3D(1.0f);
                material = new Material(0.5f, 1.0f);
            }

            public Gameobject(Vec3D p, Vec3D r, Vec3D s, Tria[] t, Material m)
            {
                position = p;
                rotation = r;
                scale = s;
                triangles = t;
                material = m;
            }
        }

        private class Vec3D
        {
            public float x, y, z;

            public Vec3D()
            {
                x = 0.0f;
                y = 0.0f;
                z = 0.0f;
            }

            public Vec3D(float a)
            {
                this.x = a;
                this.y = a;
                this.z = a;
            }

            public Vec3D(float x, float y, float z)
            {
                this.x = x;
                this.y = y;
                this.z = z;
            }

            public float Length()
            {
                return (float)Math.Sqrt(x * x + y * y + z * z);
            }

            public float LengthSquared()
            {
                return x * x + y * y + z * z;
            }

            public Vec3D Normalize()
            {
                return this / Length();
            }

            public static float Dot(Vec3D a, Vec3D b)
            {
                return (a.x * b.x + a.y * b.y + a.z * b.z);
            }

            public static Vec3D Cross(Vec3D b, Vec3D a)
            {
                return new Vec3D
                    (
                    a.y * b.z - a.z * b.y,
                    a.z * b.x - a.x * b.z,
                    a.x * b.y - a.y * b.x
                    );
            }

            public static Vec3D operator +(Vec3D a, Vec3D b)
            {
                return new Vec3D(a.x + b.x, a.y + b.y, a.z + b.z);
            }

            public static Vec3D operator -(Vec3D a, Vec3D b)
            {
                return new Vec3D(a.x - b.x, a.y - b.y, a.z - b.z);
            }

            public static Vec3D operator *(Vec3D a, float k)
            {
                return new Vec3D(a.x * k, a.y * k, a.z * k);
            }

            public static Vec3D operator /(Vec3D a, float k)
            {
                return new Vec3D(a.x / k, a.y / k, a.z / k);
            }

            public override string ToString()
            {
                return "(" + x + ", " + y + ", " + z + ")";
            }
        }

        private class Ray
        {
            public Vec3D origin, direction;
            public float length;

            public Ray(Vec3D start, Vec3D end)
            {
                origin = start;
                direction = end - start;
                length = direction.Length();
                direction = direction.Normalize();
            }

            public void Reflect(Vec3D normal)
            {
                direction = direction - normal * 2 * Vec3D.Dot(normal, direction);
            }

            public static Ray operator ++(Ray a)
            {
                return new Ray(a.origin + a.direction, a.origin + a.direction * 2.0f);
            }
        }

        private class Tria
        {
            public Vec3D normal;
            public Vec3D[] points;
            public Color color;

            public Tria(Vec3D a, Vec3D b, Vec3D c, Color col)
            {
                points = new Vec3D[3];
                points[0] = a;
                points[1] = b;
                points[2] = c;
                color = col;
                CalculateNormal();
            }

            public void CalculateNormal()
            {
                normal = Vec3D.Cross(points[0] - points[1], points[2] - points[1]).Normalize();
            }
        }

        private class Chey
        {
            public Key key;
            public bool Down, Up, Held, Raised;

            public Chey(Key key)
            {
                this.key = key;
                Down = Up = Held = false;
                Raised = true;
            }
        }

        private class DirectBitmap  : IDisposable
        {
            public Bitmap Bitmap { get; private set; }
            public int[] Bits { get; private set; }
            public bool Disposed { get; private set; }
            public int Height { get; private set; }
            public int Width { get; private set; }

            public GCHandle BitsHandle { get; private set; }

            public DirectBitmap(int width, int height)
            {
                Width = width;
                Height = height;
                Bits = new int[width * height];
                BitsHandle = GCHandle.Alloc(Bits, GCHandleType.Pinned);
                Bitmap = new Bitmap(width, height, width * 4, PixelFormat.Format32bppArgb, BitsHandle.AddrOfPinnedObject());
            }

            public void SetPixel(int x, int y, Color colour)
            {
                int index = x + (y * Width);
                int col = colour.ToArgb();

                Bits[index] = col;
            }

            public Color GetPixel(int x, int y)
            {
                int index = x + (y * Width);
                int col = Bits[index];
                Color result = Color.FromArgb(col);

                return result;
            }

            public void Dispose()
            {
                Dispose(true);
            }

            protected virtual void Dispose(bool boolean)
            {
                if (Disposed) return;
                Disposed = true;
                Bitmap.Dispose();
                BitsHandle.Free();
            }
        }

        [StructLayout(LayoutKind.Sequential, Pack = 16)]
        public struct VertexPositionTexture
        {
            public VertexPositionTexture(Vector3 position, Vector2 textureUV)
            {
                Position = new Vector4(position, 1.0f);
                TextureUV = textureUV;
                padding = new Vector2();
            }

            public Vector4 Position;
            public Vector2 TextureUV;
            private Vector2 padding;
        }
    }
}
