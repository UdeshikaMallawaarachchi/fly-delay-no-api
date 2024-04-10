(function() {
  const fn = function() {
    (function(root) {
      function now() {
        return new Date();
      }
    
      const force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
    
    const element = document.getElementById("cbfe9de3-9509-4d18-936b-932febb7b6c2");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'cbfe9de3-9509-4d18-936b-932febb7b6c2' but no matching script tag was found.")
        }
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error(url) {
          console.error("failed to load " + url);
        }
    
        for (let i = 0; i < css_urls.length; i++) {
          const url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error.bind(null, url);
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        for (let i = 0; i < js_urls.length; i++) {
          const url = js_urls[i];
          const element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error.bind(null, url);
          element.async = false;
          element.src = url;
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      const js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-3.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-gl-3.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-3.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-mathjax-3.2.1.min.js"];
      const css_urls = [];
    
      const inline_js = [    function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        function(Bokeh) {
          (function() {
            const fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                  const docs_json = '{"a6c96e34-a72f-4404-8489-6acd0366e6af":{"version":"3.2.1","title":"Bokeh Application","roots":[{"type":"object","name":"Column","id":"p1833","attributes":{"children":[{"type":"object","name":"Figure","id":"p1794","attributes":{"width":400,"height":400,"x_range":{"type":"object","name":"DataRange1d","id":"p1795"},"y_range":{"type":"object","name":"DataRange1d","id":"p1796"},"x_scale":{"type":"object","name":"LinearScale","id":"p1803"},"y_scale":{"type":"object","name":"LinearScale","id":"p1804"},"title":{"type":"object","name":"Title","id":"p1801"},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p1828","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p1791","attributes":{"selected":{"type":"object","name":"Selection","id":"p1792","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p1793"},"data":{"type":"map","entries":[["x",[1,2,3]],["y",[4,6,5]]]}}},"view":{"type":"object","name":"CDSView","id":"p1829","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p1830"}}},"glyph":{"type":"object","name":"Circle","id":"p1825","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":20},"line_color":{"type":"value","value":"#1f77b4"},"fill_color":{"type":"value","value":"#1f77b4"}}},"nonselection_glyph":{"type":"object","name":"Circle","id":"p1826","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":20},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#1f77b4"},"fill_alpha":{"type":"value","value":0.1},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Circle","id":"p1827","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"size":{"type":"value","value":20},"line_color":{"type":"value","value":"#1f77b4"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#1f77b4"},"fill_alpha":{"type":"value","value":0.2},"hatch_alpha":{"type":"value","value":0.2}}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p1802","attributes":{"tools":[{"type":"object","name":"PanTool","id":"p1815"},{"type":"object","name":"WheelZoomTool","id":"p1816"},{"type":"object","name":"BoxZoomTool","id":"p1817","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p1818","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"SaveTool","id":"p1819"},{"type":"object","name":"ResetTool","id":"p1820"},{"type":"object","name":"HelpTool","id":"p1821"}]}},"left":[{"type":"object","name":"LinearAxis","id":"p1810","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p1811","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p1812"},"major_label_policy":{"type":"object","name":"AllLabels","id":"p1813"}}}],"below":[{"type":"object","name":"LinearAxis","id":"p1805","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p1806","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p1807"},"major_label_policy":{"type":"object","name":"AllLabels","id":"p1808"}}}],"center":[{"type":"object","name":"Grid","id":"p1809","attributes":{"axis":{"id":"p1805"}}},{"type":"object","name":"Grid","id":"p1814","attributes":{"dimension":1,"axis":{"id":"p1810"}}}]}},{"type":"object","name":"Slider","id":"p1832","attributes":{"js_property_callbacks":{"type":"map","entries":[["change:value",[{"type":"object","name":"CustomJS","id":"p1831","attributes":{"code":"\\n        console.log(&#x27;Slider changed&#x27;);\\n    "}}]]]},"title":"Slider","start":0,"end":10,"value":1,"step":0.1}}]}}]}}';
                  const render_items = [{"docid":"a6c96e34-a72f-4404-8489-6acd0366e6af","roots":{"p1833":"cbfe9de3-9509-4d18-936b-932febb7b6c2"},"root_ids":["p1833"]}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    let attempts = 0;
                    const timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
    function(Bokeh) {
        }
      ];
    
      function run_inline_js() {
        for (let i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();